"""
Stage-II / prediction-only training script for EEG->motor models (EEG_pred / EEG_pred_multihead).

This script can run:
- On a single GPU (debug or small runs)
- With DDP on multiple GPUs and optionally multiple nodes

Single GPU example:
    python train_p2_ind.py --batch_size 32 --compile False

Single-node, 4-GPU DDP:
    torchrun --standalone --nproc_per_node 4 train_p2_ind.py

Two-node, 4-GPU-per-node DDP:
    # Master node (example IP 123.456.123.456)
    torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 \
             --master_addr 123.456.123.456 --master_port 1234 train_p2_ind.py

    # Worker node
    torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 \
             --master_addr 123.456.123.456 --master_port 1234 train_p2_ind.py

If the cluster does not have Infiniband, you may need:
    NCCL_IB_DISABLE=1
"""

import os
import time
from contextlib import nullcontext
from collections import OrderedDict
import argparse

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.utils import cosine_scheduler, prepare_GPP_dataset_p2, get_metrics
from utils.myaml import load_config
from models.contraDCN import EEG_pred, EEG_pred_multihead


# ---------------------------------------------------------------------
# Global runtime state
# ---------------------------------------------------------------------
master_process = None
device = None
dtype = None
ctx = None
ddp_rank = None
device_type = None
ddp = None
ddp_world_size = None
ddp_local_rank = None


def init(args):
    """Initialize device, DDP process group, random seeds, and autocast context."""
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
    torch.backends.cudnn.benchmark = True

    device_type = "cuda" if "cuda" in device else "cpu"

    if dtype == "bfloat16":
        ptdtype = torch.bfloat16
    elif dtype == "float16":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def peri_remove(eeg, ch_names=None):
    """Zero out a pre-defined set of peripheral EEG channels."""
    outer_ring_idx = [0, 3, 7, 12, 13, 14, 15, 17, 22, 26, 27, 28,
                      33, 37, 41, 44, 45, 50, 54, 56, 57]
    eeg[:, outer_ring_idx, :] = 0.0
    return eeg


def _prepare_batch_eeg(batch, model_prefix):
    """Standardize EEG tensor shape according to model requirements."""
    if model_prefix not in ["SPaRCNet", "ContraWR", "FFCL", "STtransformer", "EEGConformer", "TCN"]:
        if isinstance(batch["EEG_X"], np.ndarray):
            batch["EEG_X"] = np.expand_dims(batch["EEG_X"], axis=1)
        elif isinstance(batch["EEG_X"], torch.Tensor):
            batch["EEG_X"] = batch["EEG_X"].unsqueeze(1)

    if model_prefix in ["TCN"]:
        if isinstance(batch["EEG_X"], np.ndarray):
            batch["EEG_X"] = np.transpose(batch["EEG_X"], (0, 2, 1))
        elif isinstance(batch["EEG_X"], torch.Tensor):
            batch["EEG_X"] = batch["EEG_X"].permute(0, 2, 1)

    return batch


def evaluate(model, dataloader, metrics):
    """Run evaluation over a dataloader and compute metrics."""
    global ctx, model_name

    model.eval()
    preds = []
    gts = []
    losses = []
    log = {}

    model_prefix = model_name.split("-")[0]

    with torch.no_grad():
        for batch in dataloader:
            batch = _prepare_batch_eeg(batch, model_prefix)

            data = {
                "EEG": batch["EEG_X"].float().to(device, non_blocking=True) if "EEG_X" in batch else None,
                "domain": batch["domain"].to(device, non_blocking=True) if "domain" in batch else None,
                "Y": batch["Y"].to(device, non_blocking=True) if "Y" in batch else None,
            }
            if args.peri_remove and data["EEG"] is not None:
                data["EEG"] = peri_remove(data["EEG"], args.ch_names)

            with ctx:
                loss, logits = model(data)

            preds.append(logits.detach().cpu())
            gts.append(data["Y"].detach().cpu())
            losses.append(loss.item())

    if len(preds) == 0:
        return {f"val/{m}": 0.0 for m in metrics}

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    avg_loss = float(np.mean(losses))

    if "r2" in metrics:
        results = get_metrics(preds.numpy(), gts.numpy(), metrics, is_binary=False)
    elif "f1_weighted" not in metrics:
        results = get_metrics(torch.sigmoid(preds).numpy(), gts.numpy(), metrics, is_binary=True)
    else:
        results = get_metrics(preds.numpy(), gts.numpy(), metrics, is_binary=False)

    log["val/total_loss"] = avg_loss
    for k, v in results.items():
        log[f"val/{k}"] = v

    model.train()
    return log


def main(args):
    global master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, f"{args.dataset}_ind", model_name)
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialization mode: scratch / resume / pretrained
    # ------------------------------------------------------------------
    if args.resume:
        init_from = "resume"
        if master_process:
            print("\n\nInit from resume\n")
    elif args.pretrained:
        init_from = "pretrained"
        if master_process:
            print(f"\n\nInit from pretrained file {args.pretrained_filename}\n")
    else:
        init_from = "scratch"
        if master_process:
            print("\n\nInit from scratch\n")

    iter_num = 0
    save_filename = f"ckpt_fold{test_fold}_{val_fold}{extra_info}"
    save_filename_test = f"ckpt_fold{test_fold}{extra_info}"

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
            nFiltLaterLayer=args.nFiltLaterLayer,
        ).float().to(device)
        if master_process:
            print("Using EEG_pred (single-head).")

    start_epoch = 0
    best_val_monitor_metric = -1e9
    best_test_monitor_metric = -1e9
    best_test_metric = {}

    # ------------------------------------------------------------------
    # Load checkpoint / pretrained encoder
    # ------------------------------------------------------------------
    if init_from == "resume":
        ckpt_path = os.path.join(args.out_dir, f"{args.dataset}_ind", model_name, save_filename + ".pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        iter_num = checkpoint["iter_num"]
        start_epoch = checkpoint["epoch"] + 1
        best_val_monitor_metric = checkpoint.get("best_val_monitor_metric", best_val_monitor_metric)
        best_test_monitor_metric = checkpoint.get("best_test_monitor_metric", best_test_monitor_metric)
        if master_process:
            print(f"Resumed from {ckpt_path}")
            with open(log_txt, "w") as f:
                f.write(f"resume from {ckpt_path}\n")
                f.write(f"start_epoch: {start_epoch}\n")
    elif init_from == "pretrained":
        ckpt_path = os.path.join(args.pretrained_dir, args.pretrained_filename)
        checkpoint = torch.load(ckpt_path, map_location=device)
        pretrained_ckpt = checkpoint["model"]

        eeg_dict = OrderedDict()
        for key in list(pretrained_ckpt.keys()):
            if key.startswith("eeg_encoder."):
                eeg_dict[key[len("eeg_encoder.") :]] = pretrained_ckpt[key]

        model.encoder.load_state_dict(eeg_dict, strict=True)
        if master_process:
            print(f"Loaded encoder from {ckpt_path}")
            with open(log_txt, "w") as f:
                f.write(f"pretrained from {ckpt_path}\n")

    model.to(device)

    # ------------------------------------------------------------------
    # Dataset and dataloaders (GPP individual folds)
    # ------------------------------------------------------------------
    if master_process:
        print("Preparing dataloaders...")

    dataset_train, dataset_val, dataset_test = prepare_GPP_dataset_p2(
        args.dataset_dir, train_sessions, val_sessions, test_sessions
    )

    metrics = ["r2", "rmse", "pearsonr"]
    monitor = "r2"

    default_num_workers = min(10, max(2, (os.cpu_count() or 4) // max(1, ddp_world_size)))

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=default_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=default_num_workers > 0,
        )

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=default_num_workers,
            pin_memory=True,
            drop_last=False,
        )

        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=default_num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            num_workers=default_num_workers,
            pin_memory=False,
            drop_last=True,
            shuffle=True,
            persistent_workers=default_num_workers > 0,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=default_num_workers,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            persistent_workers=default_num_workers > 0,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=int(1.5 * args.batch_size),
            num_workers=default_num_workers,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
            persistent_workers=default_num_workers > 0,
        )

    # ------------------------------------------------------------------
    # Optimizer, scaler, compile, DDP wrapping
    # ------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )

    if args.compile:
        if master_process:
            print("Compiling the model... (this may take a while)")
            with open(log_txt, "a") as f:
                f.write("compiling the model...\n")
        model = torch.compile(model)

    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

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
        with open(log_txt, "a") as f:
            f.write("wandb activated\n")
    elif master_process:
        with open(log_txt, "a") as f:
            f.write("wandb NOT activated\n")

    num_training_steps_per_epoch = max(1, len(dataset_train) // (args.batch_size * ddp_world_size))
    lr_schedule_values = cosine_scheduler(
        args.learning_rate,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )

    raw_model = model.module if ddp else model

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        if ddp:
            data_loader_train.sampler.set_epoch(epoch)

        # Freeze encoder in the first epoch, then unfreeze
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

        for step, batch in enumerate(data_loader_train):
            batch = _prepare_batch_eeg(batch, model_name.split("-")[0])

            data = {
                "EEG": batch["EEG_X"].float().to(device, non_blocking=True) if "EEG_X" in batch else None,
                "domain": batch["domain"].to(device, non_blocking=True) if "domain" in batch else None,
                "Y": batch["Y"].to(device, non_blocking=True) if "Y" in batch else None,
            }
            if args.peri_remove and data["EEG"] is not None:
                data["EEG"] = peri_remove(data["EEG"], args.ch_names)

            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = lr

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
        # Validation and test at end of epoch
        # ------------------------------------------------------------------
        val_is_better = False

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

        log_test = evaluate(model, data_loader_test, metrics)
        log_test = {k.replace("val", "test"): v for k, v in log_test.items()}
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
            best_test_monitor_metric = log_test[monitor_metric_test]

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if master_process:
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

            if val_is_better:
                rvalue = -1.0
                checkpoint_test = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "epoch": epoch,
                    "args": vars(args),
                }
                for k, v in best_test_metric.items():
                    checkpoint_test[k] = v
                    if "pearsonr" in k:
                        rvalue = v
                torch.save(
                    checkpoint_test,
                    os.path.join(checkpoint_out_dir, save_filename_test + f"_{rvalue:.4f}.pt"),
                )

            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving periodic checkpoint {epoch} to {checkpoint_out_dir}")
                with open(log_txt, "a") as f:
                    f.write(f"saving periodic checkpoint {epoch} to {checkpoint_out_dir}\n")
                torch.save(
                    checkpoint,
                    os.path.join(checkpoint_out_dir, save_filename + f"_{epoch}.pt"),
                )

    if master_process and best_test_metric:
        msg = "Best test metrics: "
        for k, v in best_test_metric.items():
            metric_name = k.split("/")[-1]
            msg += f"{metric_name} {v:.5f}, "
        print(msg[:-2])
        with open(log_txt, "a") as f:
            f.write(msg[:-2] + "\n")

    if ddp:
        destroy_process_group()


def get_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser("Stage-II EEG prediction script", add_help=False)

    parser.add_argument("--out_dir", default="./results/", help="Output root directory.")
    parser.add_argument("--dataset_dir", default="/data/FT_dataset_2s/", help="Dataset root directory.")
    parser.add_argument(
        "--pretrained_dir",
        default="./results/GPP_ind/ContraDCN/",
        help="Directory containing pretrained Stage-I checkpoints.",
    )
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--dataset", default="GPP", help="Dataset name (only 'GPP' is used here).")

    parser.add_argument("--num_eeg_channels", type=int, default=59, help="Number of EEG channels.")
    parser.add_argument("--time_steps", type=int, default=400, help="Number of time steps in EEG input.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--motor_input_dim", type=int, default=6, help="Motor output dimension (for reference).")
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

    parser.add_argument("--wandb_log", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", default="GPP_SI")
    parser.add_argument("--wandb_runname", default="GPP_baseline")
    parser.add_argument("--wandb_api_key", type=str, default="")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--test_fold", default=9, type=int)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping threshold (0 disables).")

    parser.add_argument(
        "--decay_lr",
        default=True,
        action="store_false",
        help="Disable cosine LR schedule (enabled by default).",
    )
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    parser.add_argument("--start_name", default="1", help="Subject index for logging only.")
    parser.add_argument("--model_name", default="ContraDCN", help="Base model name.")
    parser.add_argument("--distance_type", default="cross_attention", help="'cross_attention' or 'cosine'.")

    parser.add_argument("--resume_filename", default="ckpt_-1.0000.pt", help="(unused, kept for compatibility).")
    parser.add_argument(
        "--pretrained_filename",
        default="ckpt-49.pt.pt",
        help="Filename of Stage-I checkpoint (will be overwritten in main).",
    )

    parser.add_argument("--multihead", action="store_true", help="Use EEG_pred_multihead.")
    parser.add_argument("--resume", action="store_true", help="Resume from Stage-II checkpoint.")
    parser.add_argument("--peri_remove", action="store_true", help="Zero peripheral channels during training/eval.")
    parser.add_argument("--pretrained", action="store_true", help="Initialize encoder from Stage-I checkpoint.")
    parser.add_argument("--info", default="ContraDCN", help="Information tag for this run.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name
    print("multihead:", args.multihead)

    args.info = args.info + "_" + args.distance_type
    args.wandb_runname = (
        f"{args.wandb_runname}_{args.info}_multihead"
        if args.multihead
        else f"{args.wandb_runname}_{args.info}_singlehead"
    )
    args.min_lr = args.learning_rate * 0.1

    start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    log_txt = f"./results/{model_name}_{args.start_name}_{start_time}.txt"

    # Fold indexing for GPP individual sessions
    test_fold = args.test_fold * 2 + 1
    val_fold = args.test_fold * 2
    val_idx = [i for i in range(5 * val_fold, 5 * (val_fold + 1))]
    test_idx = [i for i in range(5 * test_fold, 5 * (test_fold + 1))]

    args.ch_names = [
        "FP1", "FZ", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
        "PZ", "P3", "P7", "O1", "OZ", "O2", "P4", "P8", "CP6", "CP2",
        "CZ", "C4", "T8", "FC6", "FC2", "F4", "F8", "FP2", "AF7", "AF3",
        "AFZ", "F1", "F5", "FT7", "FC3", "C1", "C5", "TP7", "CP3", "P1",
        "P5", "PO7", "PO3", "POZ", "PO4", "PO8", "P6", "P2", "CPZ", "CP4",
        "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8", "AF4", "F2",
    ]

    train_sessions = []
    val_sessions = []
    test_sessions = []

    for idx in val_idx:
        val_sessions.append(f"{idx}_1")
        val_sessions.append(f"{idx}_2")
    for idx in test_idx:
        test_sessions.append(f"{idx}_1")
        test_sessions.append(f"{idx}_2")
    for j in range(50):
        if j not in val_idx and j not in test_idx:
            train_sessions.append(f"{j}_1")
            train_sessions.append(f"{j}_2")

    config = load_config("./config.yaml")
    config.eegnet.eeg.time_step = 400

    val_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in val_sessions]
    test_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in test_sessions]
    target_domains = val_domains + test_domains

    num_gpus = max(1, torch.cuda.device_count())
    args.batch_size = max(1, args.batch_size // num_gpus)

    extra_info = "" if args.distance_type == "cross_attention" else f"_{args.distance_type}"
    args.pretrained_filename = (
        f"ckpt_fold{val_fold}{extra_info}_contras1.0_recon1.0_align0.0_pred1.0.pt"
    )

    print("Train sessions:", train_sessions)
    print("Val sessions:  ", val_sessions)
    print("Test sessions: ", test_sessions)

    main(args)
