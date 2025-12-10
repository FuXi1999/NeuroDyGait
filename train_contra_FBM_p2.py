"""
Stage-II / prediction-only training script for EEG->motor models (EEG_pred, EEG_pred_multihead).

This script can run:
- On a single GPU (debug or small runs)
- With DDP on multiple GPUs and optionally multiple nodes

Single GPU example:
    python train_p2.py --batch_size 32 --compile False

Single-node, 4-GPU DDP:
    torchrun --standalone --nproc_per_node 4 train_p2.py

Two-node, 4-GPU-per-node DDP:
    # Master node (example IP 123.456.123.456)
    torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 \
             --master_addr 123.456.123.456 --master_port 1234 train_p2.py

    # Worker node
    torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 \
             --master_addr 123.456.123.456 --master_port 1234 train_p2.py

If the cluster does not have Infiniband, you may need:
    NCCL_IB_DISABLE=1
"""

import os
import time
from contextlib import nullcontext
import argparse
from collections import OrderedDict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.utils import (
    cosine_scheduler,
    prepare_GPP_dataset_p2,
    prepare_FBM_dataset_p2,
    get_metrics,
)
from models.contraDCN import EEG_pred, EEG_pred_multihead


# ---------------------------------------------------------------------
# Global training state (initialized in init())
# ---------------------------------------------------------------------
master_process = None
device = None
dtype = None
ctx = None
ddp = None
ddp_rank = None
ddp_world_size = None
ddp_local_rank = None
device_type = None


def init(args):
    """Initialize device, distributed environment, seeds, and autocast context."""
    global ctx, master_process, ddp, ddp_world_size, ddp_rank
    global device, dtype, device_type, ddp_local_rank

    backend = "nccl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    elif torch.cuda.is_available():
        dtype = "float16"
    else:
        dtype = "float32"

    # DDP detection: RANK is set in torchrun
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_world_size = 1
        ddp_local_rank = 0
        master_process = True
        seed_offset = 0

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"

    if dtype == "bfloat16":
        ptdtype = torch.bfloat16
    elif dtype == "float16":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def evaluate(model, dataloader, metrics):
    """
    Run evaluation on a dataloader and compute metrics.

    Returns:
        log: dict with keys "val/total_loss" and "val/<metric>".
    """
    global ctx, model_name, ddp_rank

    model.eval()
    preds = []
    gts = []
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            # Standardize EEG shape for typical models: [B, 1, C, T]
            if model_name.split("-")[0] not in [
                "SPaRCNet",
                "ContraWR",
                "FFCL",
                "STtransformer",
                "EEGConformer",
                "TCN",
            ]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.expand_dims(batch["EEG_X"], axis=1)
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].unsqueeze(1)

            # TCN expects [B, C, T]
            if model_name.split("-")[0] in ["TCN"]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.transpose(batch["EEG_X"], (0, 2, 1))
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].permute(0, 2, 1)

            data = {
                "EEG": batch["EEG_X"].float().to(device, non_blocking=True) if "EEG_X" in batch else None,
                "domain": batch["domain"].to(device, non_blocking=True) if "domain" in batch else None,
                "Y": batch["Y"].to(device, non_blocking=True) if "Y" in batch else None,
            }

            with ctx:
                loss, logits = model(data)

            preds.append(logits.detach().cpu())
            gts.append(data["Y"].detach().cpu())
            losses.append(loss.item())

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    avg_loss = float(np.mean(losses))

    # Compute metrics
    if "r2" in metrics:
        results = get_metrics(preds.numpy(), gts.numpy(), metrics, is_binary=False)
    elif "f1_weighted" not in metrics:
        results = get_metrics(torch.sigmoid(preds).numpy(), gts.numpy(), metrics, is_binary=True)
    else:
        results = get_metrics(preds.numpy(), gts.numpy(), metrics, is_binary=False)

    log = {"val/total_loss": avg_loss}
    for k, v in results.items():
        log[f"val/{k}"] = v

    model.train()
    return log


def main(args):
    global master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, args.dataset, model_name)
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        print("Preparing dataloader...")
        with open(log_txt, "w") as f:
            f.write(f"start_name: {args.start_name}\n")
            f.write(str(int(args.start_name)))
            f.write(str(val_sessions) + "\n")
            f.write(str(test_sessions) + "\n")
            f.write(str(train_sessions) + "\n")
            f.write("prepare dataloader...\n")

    # ------------------------------------------------------------------
    # Dataset-specific preparation (GPP and FBM Stage-II datasets)
    # ------------------------------------------------------------------
    if args.dataset == "GPP":
        dataset_train, dataset_val, dataset_test = prepare_GPP_dataset_p2(
            args.dataset_dir, train_sessions, val_sessions, test_sessions
        )
        val_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in val_sessions]
        test_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in test_sessions]
        target_domains = val_domains + test_domains

        if master_process:
            with open(log_txt, "a") as f:
                f.write(f"len(train): {len(dataset_train)}\n")
                f.write(f"len(val): {len(dataset_val)}\n")
                f.write(f"len(test): {len(dataset_test)}\n")

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"

    elif args.dataset == "FBM":
        dataset_train, dataset_val, dataset_test = prepare_FBM_dataset_p2(
            args.dataset_dir, train_sessions, val_sessions, test_sessions
        )
        val_domains = [int(s.split("_")[0]) - 2 for s in val_sessions]
        test_domains = [int(s.split("_")[0]) - 2 for s in test_sessions]
        target_domains = val_domains + test_domains

        if master_process:
            print("args.dataset_dir:", args.dataset_dir)
            print(f"len(train): {len(dataset_train)}")
            print(f"len(val):   {len(dataset_val)}")
            print(f"len(test):  {len(dataset_test)}")

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if master_process:
        print("Finished dataset preparation.")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
        )

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )

        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=10,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=10,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    # Init mode: scratch / resume / pretrained
    # ------------------------------------------------------------------
    if args.resume:
        init_from = "resume"
    elif args.pretrained:
        init_from = "pretrained"
    else:
        init_from = "scratch"

    if master_process:
        print(f"Initialization mode: {init_from}")
        with open(log_txt, "a") as f:
            f.write("init_from: " + init_from + "\n")

    # Filename suffix based on dataset fold and distance_type
    save_filename = f"ckpt_FBM_p2_fold{test_fold}_{val_fold}{extra_info}"
    save_filename_test = f"ckpt_FBM_p2_fold{test_fold}{extra_info}"
    save_filename_val = f"ckpt_FBM_p2_fold{val_fold}{extra_info}"

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------
    if args.multihead:
        model = EEG_pred_multihead(
            num_eeg_channels=args.num_eeg_channels,
            time_steps=args.time_steps,
            embed_dim=args.embed_dim,
            kernel_width=args.kernel_width,
            pool_width=args.pool_width,
            output_dim=args.motor_input_dim,
            nFiltLaterLayer=args.nFiltLaterLayer,
            target_domains=target_domains,
        ).float().to(device)
        if master_process:
            print("Using EEG_pred_multihead.")
    else:
        model = EEG_pred(
            num_eeg_channels=args.num_eeg_channels,
            time_steps=args.time_steps,
            embed_dim=args.embed_dim,
            kernel_width=args.kernel_width,
            pool_width=args.pool_width,
            output_dim=args.motor_input_dim,
            nFiltLaterLayer=args.nFiltLaterLayer,
        ).float().to(device)
        if master_process:
            print("Using EEG_pred (single-head).")

    iter_num = 0
    start_epoch = 0

    # ------------------------------------------------------------------
    # Load from checkpoint or pretrained encoder
    # ------------------------------------------------------------------
    if init_from == "resume":
        ckpt_path = os.path.join(args.out_dir, args.dataset, model_name, save_filename + ".pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        iter_num = checkpoint["iter_num"]
        start_epoch = checkpoint["epoch"]
        if master_process:
            print(f"Resuming from {ckpt_path}")
    elif init_from == "pretrained":
        ckpt_path = os.path.join(args.pretrained_dir, pretrained_filename)
        checkpoint = torch.load(ckpt_path, map_location=device)
        pretrained_ckpt = checkpoint["model"]

        eeg_dict = OrderedDict()
        for key in list(pretrained_ckpt.keys()):
            if key.startswith("eeg_encoder."):
                eeg_dict[key[len("eeg_encoder.") :]] = pretrained_ckpt[key]

        # At this point, model.encoder must exist
        model.encoder.load_state_dict(eeg_dict, strict=True)
        if master_process:
            print(f"Loaded encoder weights from {ckpt_path}")
            with open(log_txt, "a") as f:
                f.write("pretrained from " + ckpt_path + "\n")
                f.write(f"iter_num_pretrained: {checkpoint.get('iter_num', 'NA')}\n")
                f.write(f"epoch_pretrained: {checkpoint.get('epoch', 'NA')}\n")

    model.to(device)

    # GradScaler (only useful when dtype is float16)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )

    # Optional compilation
    if args.compile:
        if master_process:
            print("Compiling the model... (this may take a while)")
            with open(log_txt, "a") as f:
                f.write("compiling the model...\n")
        model = torch.compile(model)

    # Wrap in DDP if needed
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # Optional wandb logging
    if args.wandb_log and master_process:
        import wandb

        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_runname,
            dir=os.path.join(args.out_dir, "wandb"),
            resume=False,
        )

    # Cosine LR schedule
    num_training_steps_per_epoch = max(1, len(dataset_train) // (args.batch_size * ddp_world_size))
    lr_schedule_values = cosine_scheduler(
        args.learning_rate,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    best_val_monitor_metric = -1e9
    best_test_monitor_metric = -1e9
    best_test_metric = {}
    best_val_metric = {}

    raw_model = model.module if ddp else model

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        # Freeze encoder for the first epoch, then unfreeze
        if epoch < 1:
            if ddp:
                for p in model.module.encoder.parameters():
                    p.requires_grad = False
                model.module.encoder.eval()
            else:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                model.encoder.eval()
        else:
            if ddp:
                for p in model.module.encoder.parameters():
                    p.requires_grad = True
                model.module.encoder.train()
            else:
                for p in model.encoder.parameters():
                    p.requires_grad = True
                model.encoder.train()

        if ddp:
            # Ensure different shuffling each epoch
            data_loader_train.sampler.set_epoch(epoch)

        for step, batch in enumerate(data_loader_train):
            # Standardize EEG batch shape
            if model_name.split("-")[0] not in [
                "SPaRCNet",
                "ContraWR",
                "FFCL",
                "STtransformer",
                "EEGConformer",
                "TCN",
            ]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.expand_dims(batch["EEG_X"], axis=1)
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].unsqueeze(1)

            if model_name.split("-")[0] in ["TCN"]:
                if isinstance(batch["EEG_X"], np.ndarray):
                    batch["EEG_X"] = np.transpose(batch["EEG_X"], (0, 2, 1))
                elif isinstance(batch["EEG_X"], torch.Tensor):
                    batch["EEG_X"] = batch["EEG_X"].permute(0, 2, 1)

            data = {
                "EEG": batch["EEG_X"].float().to(device, non_blocking=True) if "EEG_X" in batch else None,
                "domain": batch["domain"].to(device, non_blocking=True) if "domain" in batch else None,
                "Y": batch["Y"].to(device, non_blocking=True) if "Y" in batch else None,
            }

            # Step-wise learning rate
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # In DDP, only sync gradients every gradient_accumulation_steps
            if ddp:
                model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

            with ctx:
                loss, log = model(data)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if (iter_num + 1) % args.log_interval == 0 and master_process:
                msg = f"Epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: "
                for k, v in log.items():
                    msg += f"{k.split('/')[-1]} {v:.4f}, "
                print(msg[:-2])
                with open(log_txt, "a") as f:
                    f.write(msg[:-2] + "\n")

                if args.wandb_log:
                    import wandb

                    log_train = log.copy()
                    log_train.update({"iter": iter_num, "lr": lr})
                    wandb.log(log_train)

            iter_num += 1

        # ------------------------------------------------------------------
        # End-of-epoch validation and testing
        # ------------------------------------------------------------------
        val_is_better = False
        test_is_better = False
        test_metric_list = []
        val_metric_list = []

        log_val = evaluate(model, data_loader_val, metrics)
        if master_process:
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write("=" * 10 + "\n")
            msg = "Evaluate EEG: "
            for k, v in log_val.items():
                msg += f"{k.split('/')[-1]} {v:.4f}, "
            print(msg[:-2])
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write(msg[:-2] + "\n")
                ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                f.write(f"end time: {ts}\n")
            if args.wandb_log:
                import wandb

                wandb.log(log_val)

        log = evaluate(model, data_loader_test, metrics)
        log_test = {k.replace("val", "test"): v for k, v in log.items()}

        if master_process:
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write("=" * 10 + "\n")
            msg = "Test EEG: "
            for k, v in log_test.items():
                msg += f"{k.split('/')[-1]} {v:.4f}, "
            print(msg[:-2])
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write(msg[:-2] + "\n")
                ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                f.write(f"end time: {ts}\n")
            if args.wandb_log:
                import wandb

                wandb.log(log_test)

        monitor_metric_val = f"val/{monitor}"
        monitor_metric_test = f"test/{monitor}"

        if log_val[monitor_metric_val] > best_val_monitor_metric:
            val_is_better = True
            best_val_monitor_metric = log_val[monitor_metric_val]
            best_test_metric = log_test.copy()

        if log_test[monitor_metric_test] > best_test_monitor_metric:
            test_is_better = True
            best_test_monitor_metric = log_test[monitor_metric_test]
            best_val_metric = log_val.copy()

        if val_is_better:
            test_metric_list.append(log_test.copy())
        if test_is_better:
            val_metric_list.append(log_val.copy())

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if master_process:
            rvalue = -1.0
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "epoch": epoch,
                "args": vars(args),
                "best_val_monitor_metric": best_val_monitor_metric,
                "best_test_monitor_metric": best_test_monitor_metric,
            }

            print(f"saving checkpoint to {checkpoint_out_dir}")
            with open(log_txt, "a") as f:
                f.write(f"saving checkpoint to {checkpoint_out_dir}\n")

            torch.save(checkpoint, os.path.join(checkpoint_out_dir, save_filename + ".pt"))

            if val_is_better and test_metric_list:
                checkpoint_test = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "epoch": epoch,
                    "args": vars(args),
                }
                for k, v in test_metric_list[-1].items():
                    checkpoint_test[k] = v
                    if "pearsonr" in k:
                        rvalue = v
                torch.save(
                    checkpoint_test,
                    os.path.join(checkpoint_out_dir, save_filename_test + f"_{rvalue:.4f}.pt"),
                )

            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                with open(log_txt, "a") as f:
                    f.write(f"saving checkpoint {epoch} to {checkpoint_out_dir}\n")
                torch.save(
                    checkpoint_test,
                    os.path.join(checkpoint_out_dir, save_filename_test + f"_{epoch}.pt"),
                )
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_out_dir, save_filename + f"_{epoch}.pt"),
                )

    if master_process and best_test_metric:
        msg = "Saved best test metrics: "
        for key, value in best_test_metric.items():
            metric_name = key.split("/")[-1]
            msg += f"{metric_name} {value:.4f}, "
        print(msg[:-2])
        with open(log_txt, "a") as f:
            f.write(msg[:-2] + "\n")

    if ddp:
        destroy_process_group()


def get_args():
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser("Stage-II EEG->motor training script", add_help=False)

    # Paths and logging
    parser.add_argument("--out_dir", default="./results/", help="Output root directory.")
    parser.add_argument("--dataset_dir", default="/data/FT_dataset_2s/", help="Dataset root directory.")
    parser.add_argument(
        "--pretrained_dir",
        default="/root/SI_GPP/results/checkpoints_400/ContraDCN/",
        help="Directory containing Stage-I pretrained models.",
    )
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--dataset", default="GPP", help="Dataset name: 'GPP' or 'FBM'.")

    # Model structure
    parser.add_argument("--num_eeg_channels", type=int, default=59, help="Number of EEG channels.")
    parser.add_argument("--time_steps", type=int, default=400, help="Number of time steps.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--motor_input_dim", type=int, default=6, help="Motor output dimension.")
    parser.add_argument("--kernel_width", type=int, default=20, help="Convolution kernel width.")
    parser.add_argument("--pool_width", type=int, default=3, help="Pooling width.")
    parser.add_argument(
        "--nFiltLaterLayer",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200],
        help="Filter sizes in later convolution layers.",
    )
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability.")
    parser.add_argument("--max_norm", type=float, default=2.0, help="Max-norm constraint for weights.")

    # wandb
    parser.add_argument("--wandb_log", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", default="GPP_SI")
    parser.add_argument("--wandb_runname", default="GPP_baseline")
    parser.add_argument("--wandb_api_key", type=str, default="")

    # Training hyperparameters
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--test_fold", default=10, type=int)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping threshold (0 disables).")

    # NOTE: default=True; passing --decay_lr will flip it to False (kept for compatibility)
    parser.add_argument(
        "--decay_lr",
        default=True,
        action="store_false",
        help="Disable cosine LR schedule (by default it is enabled).",
    )

    parser.add_argument("--seed", default=42, type=int)

    # Model / training behavior
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    parser.add_argument("--start_name", default="1", help="Subject index for logging.")
    parser.add_argument("--model_name", default="ContraDCN", help="Base model name (for logging / filenames).")
    parser.add_argument("--distance_type", default="cross_attention", help="'cross_attention' or 'cosine'.")

    parser.add_argument("--info", default="relcon_FBM", help="Tag string appended to run name.")

    parser.add_argument("--multihead", action="store_true", help="Use EEG_pred_multihead.")
    parser.add_argument("--resume", action="store_true", help="Resume from a previous Stage-II checkpoint.")
    parser.add_argument("--pretrained", action="store_true", help="Initialize encoder from Stage-I checkpoint.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name

    print("multihead:", args.multihead)

    if "FBM" not in args.info:
        args.info = args.info + "_FBM"
    args.info = args.info + "_" + args.distance_type

    args.min_lr = args.learning_rate * 0.1

    # wandb run name
    if args.multihead:
        args.wandb_runname = args.wandb_runname + args.info + "_multihead"
    else:
        args.wandb_runname = args.wandb_runname + args.info

    start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    log_txt = f"./results/{model_name}_{args.dataset}_{start_time}.txt"

    # ------------------------------------------------------------------
    # Dataset-dependent fold and session config
    # ------------------------------------------------------------------
    if args.dataset == "GPP":
        val_idx = list(range(40, 45))
        test_idx = list(range(45, 50))
        sessions = ["1", "2"]
        args.num_eeg_channels = 59
        sbjs = list(range(1, 51))
        args.motor_input_dim = 6

        # For naming consistency only (not used in splitting)
        val_fold = val_idx[0]
        test_fold = test_idx[0]

    elif args.dataset == "FBM":
        test_fold = args.test_fold
        val_fold = (args.test_fold + 6) % 9 + 2
        val_idx = [f"0{args.test_fold}"[-2:]]
        test_idx = [f"0{args.test_fold}"[-2:]]
        sessions = ["01"]
        sbjs = ["02", "03", "04", "05", "06", "07", "08", "09", "10"]
        args.num_eeg_channels = 59
        args.motor_input_dim = 8
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Distance-type info suffix for checkpoint naming
    extra_info = "" if args.distance_type == "cross_attention" else f"_{args.distance_type}"
    pretrained_filename = f"ckpt_fold{val_fold}{extra_info}_contras1.0_recon1.0_align0.0_pred1.0.pt"

    # Build train/val/test session lists
    train_sessions = []
    val_sessions = []
    test_sessions = []

    val_idx_set = set(val_idx)
    test_idx_set = set(test_idx)

    for idx in val_idx:
        for session in sessions:
            val_sessions.append(f"{idx}_{session}")
    for idx in test_idx:
        for session in sessions:
            test_sessions.append(f"{idx}_{session}")
    for j in sbjs:
        # For FBM, j is string; for GPP, j is int, so we convert to str before comparing
        j_str = str(j)
        if j_str in val_idx_set or j_str in test_idx_set:
            continue
        for session in sessions:
            train_sessions.append(f"{j_str}_{session}")

    # Split total batch size across GPUs on the node
    num_gpus = max(1, torch.cuda.device_count())
    args.batch_size = max(1, args.batch_size // num_gpus)

    print("Train sessions:", train_sessions)
    print("Val sessions:  ", val_sessions)
    print("Test sessions: ", test_sessions)

    main(args)
