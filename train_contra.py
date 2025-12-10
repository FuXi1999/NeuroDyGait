"""
Training script for EEG-motor contrastive model (ContraDCN).

Supports:
- Single GPU:
    python train.py --batch_size 32 --compile False

- Single-node multi-GPU DDP:
    torchrun --standalone --nproc_per_node 4 train.py

- Multi-node multi-GPU DDP (example: 2 nodes, 4 GPUs per node):
  # Master node (assume IP = 123.456.123.456)
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 \
           --master_addr 123.456.123.456 --master_port 1234 train.py

  # Worker node:
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 \
           --master_addr 123.456.123.456 --master_port 1234 train.py

If the cluster does not have Infiniband, you may need:
    NCCL_IB_DISABLE=1
"""

import os
import time
from contextlib import nullcontext
import argparse

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.utils import cosine_scheduler, prepare_GPP_dataset, get_metrics
from models.contraDCN import EEGMotorContrastiveModel, EEGMotorContrastiveModel_multihead

# ------------------------------------------------------------------
# Global state, initialized in init()
# ------------------------------------------------------------------
master_process = None         # True only on rank 0 (for logging and checkpointing)
device = None
dtype = None                  # 'float32' / 'bfloat16' / 'float16'
ctx = None                    # autocast context
ddp = None
ddp_rank = None
ddp_world_size = None
ddp_local_rank = None
device_type = None            # 'cuda' or 'cpu'


def init(args):
    """Initialize device, distributed environment, seeds, and autocast context."""
    global ctx, master_process, ddp, ddp_world_size, ddp_rank
    global device, dtype, device_type, ddp_local_rank

    backend = "nccl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select base dtype; autocast dtype is configured separately
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = "bfloat16"
    elif torch.cuda.is_available():
        dtype = "float16"
    else:
        dtype = "float32"

    # ------------------------------------------------------------------
    # DDP initialization
    # ------------------------------------------------------------------
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

    # Set random seed
    torch.manual_seed(args.seed + seed_offset)

    # Allow TF32 for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"

    # Configure autocast dtype
    if dtype == "bfloat16":
        ptdtype = torch.bfloat16
    elif dtype == "float16":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def peri_remove(eeg, ch_names=None):
    """
    Zero out outer-ring channels to suppress peripheral noise.

    eeg: [..., C, T] where C is the channel dimension.
    ch_names: optional channel name list (not used, kept for compatibility).
    """
    # Indices of outer-ring EEG channels, based on args.ch_names
    outer_ring_idx = [0, 3, 7, 12, 13, 14, 15, 17, 22, 26, 27,
                      28, 33, 37, 41, 44, 45, 50, 54, 56, 57]

    eeg[..., outer_ring_idx, :] = 0.0
    return eeg


def evaluate(model, dataloader, metrics, args, model_name):
    """
    Evaluate the model on a given dataloader and compute metrics.

    Returns:
        log: dict mapping "val/metric_name" -> value
    """
    global ctx, ddp_rank
    model.eval()

    preds = []
    gts = []
    total_loss = []

    with torch.no_grad():
        for batch in dataloader:
            # Standardize EEG input to [B, 1, C, T] for most models
            if model_name.split("-")[0] not in [
                "SPaRCNet", "ContraWR", "FFCL", "STtransformer", "EEGConformer", "TCN"
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

            # Optional peripheral channel removal
            if args.peri_remove and data["EEG"] is not None:
                # data["EEG"]: [B, 1, C, T] -> [B*1, C, T] for peri_remove
                b, c, ch, t = data["EEG"].shape
                eeg_flat = data["EEG"].view(b * c, ch, t)
                eeg_flat = peri_remove(eeg_flat, args.ch_names)
                data["EEG"] = eeg_flat.view(b, c, ch, t)

            with ctx:
                loss, logits = model(data)

            preds.append(logits.detach().cpu())
            gts.append(data["Y"].detach().cpu())
            total_loss.append(loss.item())

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    avg_loss = float(np.mean(total_loss))

    # Optionally save predictions on rank 0 for offline analysis
    if ddp_rank == 0:
        _tmp = torch.stack([preds, gts], dim=1)
        # Example:
        # np.save(f"predntrue_{_tmp.shape[0]}.npy", _tmp.numpy())

    # Metric computation
    if "r2" in metrics:
        results = get_metrics(preds.numpy(), gts.numpy(), metrics, is_binary=False)
    elif "f1_weighted" not in metrics:
        results = get_metrics(torch.sigmoid(preds).numpy(), gts.numpy(), metrics, is_binary=True)
    else:
        results = get_metrics(preds.numpy(), gts.numpy(), metrics, is_binary=False)

    log = {f"val/{k}": v for k, v in results.items()}
    log["val/total_loss"] = avg_loss

    model.train()
    return log


def main(args):
    global master_process

    init(args)

    # ------------------------------------------------------------------
    # Paths and logging
    # ------------------------------------------------------------------
    model_name = args.model_name
    checkpoint_out_dir = os.path.join(args.out_dir, "checkpoints_400", model_name)

    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        print("Preparing dataloader...")
        with open(log_txt, "w") as f:
            f.write(f"start_name: {args.start_name}\n")
            f.write(f"{int(args.start_name)}\n")
            f.write(str(val_sessions) + "\n")
            f.write(str(test_sessions) + "\n")
            f.write(str(train_sessions) + "\n")
            f.write("prepare dataloader...\n")

    # ------------------------------------------------------------------
    # Dataset preparation (currently only supports GPP)
    # ------------------------------------------------------------------
    if args.dataset == "GPP":
        dataset_train, dataset_val, dataset_test = prepare_GPP_dataset(
            args.dataset_dir, train_sessions, val_sessions, test_sessions
        )

        val_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in val_sessions]
        test_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in test_sessions]
        target_domains = val_domains + test_domains

        print(f"len(train): {len(dataset_train)}")
        print(f"len(val):   {len(dataset_val)}")
        print(f"len(test):  {len(dataset_test)}")

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print("Finished dataset preparation.")
    with open(log_txt, "a") as f:
        f.write("finished dataloader!\n")

    # ------------------------------------------------------------------
    # DataLoaders and samplers
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
    # Model construction
    # ------------------------------------------------------------------
    extra_info = "" if args.distance_type == "cross_attention" else f"_{args.distance_type}"
    save_filename = (
        f"ckpt_fold{val_fold}_{test_fold}"
        f"{extra_info}_contras{args.warmup_contrastive_weight}"
        f"_recon{args.warmup_reconstruction_weight}"
        f"_align{args.warmup_alignment_weight}"
        f"_pred{args.warmup_prediction_weight}"
    )

    if args.multihead:
        model = EEGMotorContrastiveModel_multihead(
            num_eeg_channels=args.num_eeg_channels,
            time_steps=args.time_steps,
            embed_dim=args.embed_dim,
            kernel_width=args.kernel_width,
            pool_width=args.pool_width,
            nFiltLaterLayer=args.nFiltLaterLayer,
            dropout_p=args.dropout_p,
            max_norm=args.max_norm,
            contrastive_margin=args.contrastive_margin,
            n_domains=args.n_domains,
            target_domains=target_domains,
        ).float().to(device)
        if master_process:
            print("Using multi-head ContraDCN.")
    else:
        model = EEGMotorContrastiveModel(
            num_eeg_channels=args.num_eeg_channels,
            time_steps=args.time_steps,
            embed_dim=args.embed_dim,
            kernel_width=args.kernel_width,
            pool_width=args.pool_width,
            nFiltLaterLayer=args.nFiltLaterLayer,
            dropout_p=args.dropout_p,
            max_norm=args.max_norm,
            contrastive_margin=args.contrastive_margin,
            sparsemax=args.sparsemax,
            target_domains=target_domains,
            distance_type=args.distance_type,
        ).float().to(device)
        if master_process:
            print("Using single-head ContraDCN.")

    # ------------------------------------------------------------------
    # Resume or start from scratch
    # ------------------------------------------------------------------
    if args.resume:
        with open(log_txt, "a") as f:
            f.write("try to resume from checkpoint\n")

        ckpt_path = os.path.join(checkpoint_out_dir, save_filename + ".pt")
        if os.path.exists(ckpt_path):
            init_from = "resume"
            with open(log_txt, "a") as f:
                f.write(f"resume from checkpoint {save_filename + '.pt'}\n")
        else:
            init_from = "scratch"
            with open(log_txt, "a") as f:
                f.write(f"not found {save_filename + '.pt'}\nresume from scratch\n")
    else:
        init_from = "scratch"

    with open(log_txt, "a") as f:
        f.write("init_from: " + init_from + "\n")

    iter_num = 0
    save_filename_test = (
        f"ckpt_fold{test_fold}{extra_info}_contras{args.warmup_contrastive_weight}"
        f"_recon{args.warmup_reconstruction_weight}"
        f"_align{args.warmup_alignment_weight}"
        f"_pred{args.warmup_prediction_weight}"
    )
    save_filename_val = (
        f"ckpt_fold{val_fold}{extra_info}_contras{args.warmup_contrastive_weight}"
        f"_recon{args.warmup_reconstruction_weight}"
        f"_align{args.warmup_alignment_weight}"
        f"_pred{args.warmup_prediction_weight}"
    )

    if init_from == "resume":
        checkpoint = torch.load(os.path.join(checkpoint_out_dir, save_filename + ".pt"), map_location=device)
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint["iter_num"]
        start_epoch = checkpoint["epoch"] + 1
        with open(log_txt, "a") as f:
            f.write(f"resume from {os.path.join(checkpoint_out_dir, save_filename + '.pt')}\n")
            f.write(f"start_epoch: {start_epoch}\n")
    else:
        start_epoch = 0
        checkpoint = None

    # GradScaler is only useful when training with float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    if init_from == "resume" and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Optional torch.compile
    if args.compile:
        if master_process:
            print("Compiling the model... (this may take a while)")
            with open(log_txt, "a") as f:
                f.write("Compiling the model...\n")
        model = torch.compile(model)  # requires PyTorch >= 2.0

    # Wrap with DDP if needed
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # wandb logging
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

    # LR schedule
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
    if init_from == "resume" and checkpoint is not None:
        best_val_monitor_metric = checkpoint.get("best_val_monitor_metric", best_val_monitor_metric)
        best_test_monitor_metric = checkpoint.get("best_test_monitor_metric", best_test_monitor_metric)

    checkpoint = None  # free memory
    raw_model = model.module if ddp else model

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        # Ratio is a sampling hyperparameter used inside the model
        if epoch < 2:
            ratio = 5
        else:
            ratio = min(epoch * 2, args.batch_size)

        if ddp:
            # Ensure each epoch sees a different data partition
            data_loader_train.sampler.set_epoch(epoch)

        for step, batch in enumerate(data_loader_train):
            # Standardize EEG input to [B, 1, C, T]
            if model_name.split("-")[0] not in [
                "SPaRCNet", "ContraWR", "FFCL", "STtransformer", "EEGConformer", "TCN"
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

            # Optional peripheral channel removal
            if args.peri_remove and data["EEG"] is not None:
                b, c, ch, t = data["EEG"].shape
                eeg_flat = data["EEG"].view(b * c, ch, t)
                eeg_flat = peri_remove(eeg_flat, args.ch_names)
                data["EEG"] = eeg_flat.view(b, c, ch, t)

            # Per-iteration learning rate
            if args.decay_lr:
                lr = lr_schedule_values[iter_num]
            else:
                lr = args.learning_rate

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # In DDP, we only need to sync gradients on the last accumulation step
            if ddp:
                model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

            # Configure loss weights for warmup vs full training
            with ctx:
                if epoch < args.frozen_epoch:
                    contrastive_weight = args.warmup_contrastive_weight
                    reconstruction_weight = args.warmup_reconstruction_weight
                    prediction_weight = args.warmup_prediction_weight
                    alignment_weight = args.warmup_alignment_weight
                    loss_cross_weight = args.warmup_loss_cross_weight
                else:
                    contrastive_weight = args.contrastive_weight
                    reconstruction_weight = args.reconstruction_weight
                    prediction_weight = args.prediction_weight
                    alignment_weight = args.alignment_weight
                    loss_cross_weight = args.loss_cross_weight
                    lr = args.p2_learning_rate
                    checkpoint_out_dir = os.path.join(args.out_dir, "checkpoints_400", model_name + "_pred")

                loss, log = model(
                    data,
                    ratio=ratio,
                    contrastive_weight=contrastive_weight,
                    reconstruction_weight=reconstruction_weight,
                    prediction_weight=prediction_weight,
                    alignment_weight=alignment_weight,
                    loss_cross_weight=loss_cross_weight,
                )
                loss = loss / args.gradient_accumulation_steps

            # Backward pass with gradient scaling (if float16)
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Logging
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
                    log_train.update({"iter": iter_num, "lr": lr, "epoch": epoch})
                    wandb.log(log_train)

            iter_num += 1

        # ------------------------------------------------------------------
        # End-of-epoch evaluation on val and test
        # ------------------------------------------------------------------
        log_val = evaluate(model, data_loader_val, metrics, args, model_name)
        log_test_raw = evaluate(model, data_loader_test, metrics, args, model_name)
        log_test = {k.replace("val", "test"): v for k, v in log_test_raw.items()}

        if master_process:
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write("=" * 10 + "\n")

            msg = "Evaluate EEG: "
            for k, v in log_val.items():
                msg += f"{k.split('/')[-1]} {v:.4f}, "
            print(msg[:-2])
            with open(log_txt, "a") as f:
                f.write(msg[:-2] + "\n")

            if args.wandb_log:
                import wandb
                wandb.log(log_val)

            print("=" * 10)
            msg = "Test EEG: "
            for k, v in log_test.items():
                msg += f"{k.split('/')[-1]} {v:.4f}, "
            print(msg[:-2])
            with open(log_txt, "a") as f:
                f.write(msg[:-2] + "\n")

            if args.wandb_log:
                wandb.log(log_test)

        # Monitor metric (r2 by default)
        monitor_metric_val = f"val/{monitor}"
        monitor_metric_test = f"test/{monitor}"

        val_is_better = log_val[monitor_metric_val] > best_val_monitor_metric
        test_is_better = log_test[monitor_metric_test] > best_test_monitor_metric

        if val_is_better:
            best_val_monitor_metric = log_val[monitor_metric_val]
        if test_is_better:
            best_test_monitor_metric = log_test[monitor_metric_test]

        # ------------------------------------------------------------------
        # Checkpoint saving
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
                f.write(f"saving checkpoint to {checkpoint_out_dir} as {save_filename}.pt\n")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, save_filename + ".pt"))

            # When validation improves, save test metrics snapshot
            if val_is_better:
                checkpoint_test = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "epoch": epoch,
                    "args": vars(args),
                    **log_test,
                }
                torch.save(checkpoint_test, os.path.join(checkpoint_out_dir, save_filename_test + ".pt"))

            # When test improves, save validation metrics snapshot
            if test_is_better:
                checkpoint_val = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "epoch": epoch,
                    "args": vars(args),
                    **log_val,
                }
                torch.save(checkpoint_val, os.path.join(checkpoint_out_dir, save_filename_val + ".pt"))

            # Periodic epoch-tagged checkpoints
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                with open(log_txt, "a") as f:
                    f.write(f"saving checkpoint {epoch} to {checkpoint_out_dir}\n")

                torch.save(checkpoint, os.path.join(checkpoint_out_dir, save_filename + f"_{epoch}.pt"))
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, save_filename_test + f"_{epoch}.pt"))
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, save_filename_val + f"_{epoch}.pt"))

    if ddp:
        destroy_process_group()


def get_args():
    """Define CLI arguments, keeping only those actually used by this script."""
    parser = argparse.ArgumentParser("ContraDCN training script", add_help=False)

    # Paths and logging
    parser.add_argument("--out_dir", default="./results/", help="Output directory for logs and checkpoints.")
    parser.add_argument("--dataset_dir", default="Z:/Datasets/GPP/FT_dataset_phase/", help="GPP dataset root path.")
    parser.add_argument("--log_interval", default=10, type=int, help="Steps between logging to stdout and file.")
    parser.add_argument("--wandb_log", default=True, action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", default="GPP_contra")
    parser.add_argument("--wandb_runname", default="T4_2")
    parser.add_argument("--wandb_api_key", type=str, default="")

    # Data and model architecture
    parser.add_argument("--dataset", default="GPP", help="Dataset name (currently only supports GPP).")
    parser.add_argument("--num_eeg_channels", type=int, default=59)
    parser.add_argument("--time_steps", type=int, default=400)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--kernel_width", type=int, default=20)
    parser.add_argument("--pool_width", type=int, default=3)
    parser.add_argument("--nFiltLaterLayer", nargs="+", type=int, default=[25, 50, 100, 200])
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--max_norm", type=float, default=2.0)
    parser.add_argument("--contrastive_margin", type=float, default=1.0)
    parser.add_argument("--n_domains", type=int, default=100)

    # Training hyperparameters
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--test_fold", default=9, type=int)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Clip gradients at this value (0.0 to disable).")

    # NOTE: keeping original semantics:
    # default = True, and passing --decay_lr will set it to False.
    parser.add_argument(
        "--decay_lr",
        default=True,
        action="store_false",
        help="Disable cosine LR schedule (default: decay is enabled).",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--frozen_epoch", default=100, type=int)

    # Compilation / model settings
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on the model.")
    parser.add_argument("--start_name", default="50", help="Subject starting ID (for logging only).")
    parser.add_argument("--model_name", default="ContraDCN", help="Model name.")
    parser.add_argument("--distance_type", default="cross_attention", help="Distance type: 'cross_attention' or 'cosine'.")

    # Warmup loss weights
    parser.add_argument("--warmup_contrastive_weight", default=1.0, type=float)
    parser.add_argument("--warmup_reconstruction_weight", default=1.0, type=float)
    parser.add_argument("--warmup_prediction_weight", default=5.0, type=float)
    parser.add_argument("--warmup_alignment_weight", default=100.0, type=float)
    parser.add_argument("--warmup_loss_cross_weight", default=1.0, type=float)

    # Main loss weights after warmup
    parser.add_argument("--contrastive_weight", default=0.0, type=float)
    parser.add_argument("--reconstruction_weight", default=1.0, type=float)
    parser.add_argument("--prediction_weight", default=1.0, type=float)
    parser.add_argument("--alignment_weight", default=0.0, type=float)
    parser.add_argument("--loss_cross_weight", default=1.0, type=float)

    # Extra info
    parser.add_argument(
        "--info",
        default="relcon_contra100_pred0.5_softmax",
        help="Extra string appended to run name for easier experiment tracking.",
    )
    parser.add_argument("--p2_learning_rate", default=5e-6, type=float)

    # Multi-head / attention variants
    parser.add_argument("--multihead", action="store_true", help="Use multi-head variant of ContraDCN.")
    parser.add_argument("--sparsemax", action="store_true", help="Use sparsemax instead of softmax in attention.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument("--peri_remove", action="store_true", help="Zero out outer-ring EEG channels.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name

    # Build wandb run name: base + info + head type + attention type
    args.info = args.info + "_" + args.distance_type
    suffix = "multihead" if args.multihead else "singlehead"
    attn_type = "sparsemax" if args.sparsemax else "softmax"
    args.wandb_runname = f"{args.wandb_runname}_{args.info}_{suffix}_{attn_type}"

    start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    log_txt = f"./results/{model_name}_{args.start_name}_{start_time}.txt"

    # Fold -> val/test indices
    test_fold = args.test_fold * 2 + 1
    val_fold = args.test_fold * 2
    val_idx = list(range(5 * val_fold, 5 * (val_fold + 1)))
    test_idx = list(range(5 * test_fold, 5 * (test_fold + 1)))

    # Construct session lists
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

    # If needed, you can override with a tiny subset for debugging:
    # val_sessions = ["0_1"]
    # test_sessions = ["0_1"]
    # train_sessions = ["0_2"]

    # Match original behavior: min_lr is derived from learning_rate
    args.min_lr = args.learning_rate * 0.1

    # Split total batch size across GPUs on the current node
    num_gpus = max(1, torch.cuda.device_count())
    args.batch_size = max(1, args.batch_size // num_gpus)

    # Channel names used by peri_remove (indexing matches args.ch_names order)
    args.ch_names = [
        "FP1", "FZ", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
        "PZ", "P3", "P7", "O1", "OZ", "O2", "P4", "P8", "CP6", "CP2",
        "CZ", "C4", "T8", "FC6", "FC2", "F4", "F8", "FP2", "AF7", "AF3",
        "AFZ", "F1", "F5", "FT7", "FC3", "C1", "C5", "TP7", "CP3", "P1",
        "P5", "PO7", "PO3", "POZ", "PO4", "PO8", "P6", "P2", "CPZ", "CP4",
        "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8", "AF4", "F2",
    ]

    print("Train sessions:", train_sessions)
    print("Val sessions:  ", val_sessions)
    print("Test sessions: ", test_sessions)

    main(args)
