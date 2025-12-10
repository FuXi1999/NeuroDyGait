"""
Training script for EEG-motor contrastive model (ContraDCN) on MoBI / FBM.

Supports:
- Single GPU:
    python train_mobi_fbm.py --batch_size 32 --compile False

- Single-node multi-GPU DDP:
    torchrun --standalone --nproc_per_node 4 train_mobi_fbm.py

- Multi-node multi-GPU DDP (example: 2 nodes, 4 GPUs per node):
  # Master node (assume IP = 123.456.123.456)
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 \
           --master_addr 123.456.123.456 --master_port 1234 train_mobi_fbm.py

  # Worker node:
  torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 \
           --master_addr 123.456.123.456 --master_port 1234 train_mobi_fbm.py

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

from utils.utils import cosine_scheduler, prepare_MoBI_dataset, prepare_FBM_dataset, get_metrics
from models.contraDCN import EEGMotorContrastiveModel, EEGMotorContrastiveModel_multihead

# ------------------------------------------------------------------
# Global state, initialized in init()
# ------------------------------------------------------------------
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
    """Initialize device, distributed environment, seed, and autocast context."""
    global ctx, master_process, ddp, ddp_world_size, ddp_rank
    global device, dtype, device_type, ddp_local_rank

    backend = "nccl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Choose base training dtype; autocast dtype is configured separately
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

    # Seed
    torch.manual_seed(args.seed + seed_offset)

    # Allow TF32 on supported hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"

    # Autocast dtype
    if dtype == "bfloat16":
        ptdtype = torch.bfloat16
    elif dtype == "float16":
        ptdtype = torch.float16
    else:
        ptdtype = torch.float32

    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def evaluate(model, dataloader, metrics, save_predntrue=False):
    """
    Evaluate the model on a given dataloader and compute metrics.

    Args:
        model: PyTorch model, possibly wrapped in DDP.
        dataloader: validation or test DataLoader.
        metrics: list of metric names for get_metrics.
        save_predntrue: if True, rank 0 saves a pred+true .npy file.

    Returns:
        log: dict with keys "val/metric_name" and "val/total_loss".
    """
    global ctx, model_name, ddp_rank

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

            with ctx:
                loss, logits = model(data)

            preds.append(logits.detach().cpu())
            gts.append(data["Y"].detach().cpu())
            total_loss.append(loss.item())

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    avg_loss = float(np.mean(total_loss))

    # Optionally save predictions and labels on rank 0
    if ddp_rank == 0 and save_predntrue:
        tmp = torch.stack([preds, gts], dim=1)
        np.save(f"predntrue_{tmp.shape[0]}.npy", tmp.numpy())
        print(f"Rank 0 saved predntrue_{tmp.shape[0]}.npy")

    # Metric computation
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
    # Dataset preparation (MoBI / FBM)
    # ------------------------------------------------------------------
    if args.dataset == "MoBI":
        dataset_train, dataset_val, dataset_test = prepare_MoBI_dataset(
            args.dataset_dir, train_sessions, val_sessions, test_sessions
        )
        val_domains = [int(s.split("_")[0]) * 3 + int(s.split("_")[1]) for s in val_sessions]
        test_domains = [int(s.split("_")[0]) * 3 + int(s.split("_")[1]) for s in test_sessions]
        target_domains = val_domains + test_domains

        if master_process:
            with open(log_txt, "a") as f:
                f.write(f"len(train): {len(dataset_train)}\n")
                f.write(f"len(val): {len(dataset_val)}\n")
                f.write(f"len(test): {len(dataset_test)}\n")

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"

    elif args.dataset == "FBM":
        dataset_train, dataset_val, dataset_test = prepare_FBM_dataset(
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
            print("target_domains:", target_domains)

        metrics = ["r2", "rmse", "pearsonr"]
        monitor = "r2"

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if master_process:
        print("Finished dataset preparation.")

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
            motor_input_dim=args.motor_input_dim,
            nFiltLaterLayer=args.nFiltLaterLayer,
            dropout_p=args.dropout_p,
            max_norm=args.max_norm,
            contrastive_margin=args.contrastive_margin,
            n_domains=args.n_domains,
            target_domains=target_domains,
            distance_type=args.distance_type,
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
            motor_input_dim=args.motor_input_dim,
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

        # Optional: load pretrained encoder weights (e.g., Stage I from a source dataset)
        if args.pretrained:
            checkpoint = torch.load(args.pretrained_model_path, map_location=device)
            state_dict = checkpoint["model"]
            model_dict = model.state_dict()

            # Only load encoder layers (keys starting with 'eeg_encoder')
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and k.startswith("eeg_encoder")
            }
            model_dict.update(filtered_state_dict)
            model.load_state_dict(model_dict)

            if master_process:
                print(f"Loaded pretrained encoder weights for {len(filtered_state_dict)} layers.")

    # ------------------------------------------------------------------
    # Resume or start from scratch
    # ------------------------------------------------------------------
    if args.resume and os.path.exists(os.path.join(checkpoint_out_dir, save_filename + ".pt")):
        init_from = "resume"
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

    best_val_monitor_metric = -1e9
    best_test_monitor_metric = -1e9

    checkpoint = None
    if init_from == "resume":
        ckpt_path = os.path.join(checkpoint_out_dir, save_filename + ".pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint.get("iter_num", 0)
        start_epoch = checkpoint["epoch"] + 1
        best_val_monitor_metric = checkpoint.get("best_val_monitor_metric", best_val_monitor_metric)
        best_test_monitor_metric = checkpoint.get("best_test_monitor_metric", best_test_monitor_metric)

        with open(log_txt, "a") as f:
            f.write(f"resume from {ckpt_path}\n")
            f.write(f"start_epoch: {start_epoch}\n")
    else:
        start_epoch = 0

    model.to(device)

    # GradScaler is only useful when training with float16
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # Optimizer
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type
    )
    if init_from == "resume" and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free memory

    # Optional torch.compile
    if args.compile:
        if master_process:
            print("Compiling the model... (this may take a while)")
            with open(log_txt, "a") as f:
                f.write("Compiling the model...\n")
        model = torch.compile(model)

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
            # Ensure different shuffling across epochs
            data_loader_train.sampler.set_epoch(epoch)

        for step, batch in enumerate(data_loader_train):
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

            # Learning rate for this iteration
            if args.decay_lr:
                lr = lr_schedule_values[iter_num]
            else:
                lr = args.learning_rate

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # In DDP, sync gradients only at the last accumulation step
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

                    # Switch output directory for Stage II (prediction-focused)
                    checkpoint_out_dir_stage2 = os.path.join(
                        args.out_dir, args.dataset, model_name + "_pred"
                    )
                    if not os.path.exists(checkpoint_out_dir_stage2):
                        os.makedirs(checkpoint_out_dir_stage2, exist_ok=True)
                        if master_process:
                            print(f"Created Stage II output dir: {checkpoint_out_dir_stage2}")
                    checkpoint_out_dir = checkpoint_out_dir_stage2

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

            # Backward with gradient scaling (if enabled)
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
        # End-of-epoch evaluation
        # ------------------------------------------------------------------
        log_val = evaluate(model, data_loader_val, metrics)
        log_test_raw = evaluate(model, data_loader_test, metrics)
        log_test = {k.replace("val", "test"): v for k, v in log_test_raw.items()}

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
                tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                f.write(f"end time: {tmp_time}\n")

            if args.wandb_log:
                import wandb
                wandb.log(log_val)

        if master_process:
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write("=" * 10 + "\n")

        monitor_metric_val = f"val/{monitor}"
        monitor_metric_test = f"test/{monitor}"

        val_is_better = log_val[monitor_metric_val] > best_val_monitor_metric
        test_is_better = log_test[monitor_metric_test] > best_test_monitor_metric

        if val_is_better:
            best_val_monitor_metric = log_val[monitor_metric_val]
        if test_is_better:
            best_test_monitor_metric = log_test[monitor_metric_test]

        if master_process:
            msg = "Test EEG: "
            for k, v in log_test.items():
                msg += f"{k.split('/')[-1]} {v:.4f}, "
            print(msg[:-2])
            print("=" * 10)
            with open(log_txt, "a") as f:
                f.write(msg[:-2] + "\n")
                tmp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                f.write(f"end time: {tmp_time}\n")

            if args.wandb_log:
                import wandb
                wandb.log(log_test)

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

            # Save snapshot when validation improves
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

            # Periodic epoch-tagged checkpoints
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                with open(log_txt, "a") as f:
                    f.write(f"saving checkpoint {epoch} to {checkpoint_out_dir}\n")

                torch.save(checkpoint, os.path.join(checkpoint_out_dir, save_filename + f"_{epoch}.pt"))
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, save_filename_test + f"_{epoch}.pt"))

    if ddp:
        destroy_process_group()


def get_args():
    """Define CLI arguments, only keeping those used by this script."""
    parser = argparse.ArgumentParser("ContraDCN training script (MoBI/FBM)", add_help=False)

    # Paths and logging
    parser.add_argument("--out_dir", default="./results/", help="Output directory for logs and checkpoints.")
    parser.add_argument("--dataset_dir", default="/data/FT_dataset_2s/", help="Dataset root path.")
    parser.add_argument("--log_interval", default=10, type=int, help="Steps between logging to stdout/file.")
    parser.add_argument("--dataset", default="MoBI", help="Dataset name: 'MoBI' or 'FBM'.")

    # Data and model architecture
    parser.add_argument("--num_eeg_channels", type=int, default=60, help="Number of EEG channels.")
    parser.add_argument("--time_steps", type=int, default=400, help="Number of time steps.")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--motor_input_dim", type=int, default=6, help="Motor input dimension.")
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
    parser.add_argument("--contrastive_margin", type=float, default=1.0, help="Margin for contrastive loss.")
    parser.add_argument("--n_domains", type=int, default=100, help="Number of training domains (e.g., subjects).")

    # wandb
    parser.add_argument("--wandb_log", default=False, action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", default="MoBI_SI")
    parser.add_argument("--wandb_runname", default="MoBI_baseline")
    parser.add_argument("--wandb_api_key", type=str, default="")

    # Training hyperparameters
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_epochs", default=2, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)
    parser.add_argument("--test_fold", default=9, type=int, help="Fold index (only used for FBM).")

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping threshold (0.0 disables).")

    # NOTE: default=True, passing --decay_lr will set it False (kept for backward compatibility)
    parser.add_argument(
        "--decay_lr",
        default=True,
        action="store_false",
        help="Disable cosine LR schedule (by default it is enabled).",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--frozen_epoch", default=20, type=int, help="Epoch at which to switch to Stage II.")

    # Compilation / model settings
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    parser.add_argument("--start_name", default="1", help="Subject index start (for logging only).")
    parser.add_argument("--model_name", default="ContraDCN", help="Model name.")
    parser.add_argument("--distance_type", default="cross_attention", help="'cross_attention' or 'cosine'.")

    # Warmup loss weights
    parser.add_argument("--warmup_contrastive_weight", default=1.0, type=float)
    parser.add_argument("--warmup_reconstruction_weight", default=5.0, type=float)
    parser.add_argument("--warmup_prediction_weight", default=0.1, type=float)
    parser.add_argument("--warmup_alignment_weight", default=10.0, type=float)
    parser.add_argument("--warmup_loss_cross_weight", default=2.0, type=float)

    # Main loss weights (Stage II)
    parser.add_argument("--contrastive_weight", default=0.0, type=float)
    parser.add_argument("--reconstruction_weight", default=0.0, type=float)
    parser.add_argument("--prediction_weight", default=1.0, type=float)
    parser.add_argument("--alignment_weight", default=0.5, type=float)
    parser.add_argument("--loss_cross_weight", default=2.0, type=float)

    parser.add_argument("--info", default="relcon", help="Extra tag appended to run names.")
    parser.add_argument("--p2_learning_rate", default=5e-6, type=float, help="Stage II learning rate.")

    parser.add_argument("--multihead", action="store_true", help="Use multi-head variant.")
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained encoder weights.")
    parser.add_argument("--sparsemax", action="store_true", help="Use sparsemax attention.")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")

    parser.add_argument(
        "--pretrained_model_path",
        default="/home/intern/T4/SI_GPP/results/checkpoints_400/ContraDCN/ckpt_relcon_contra100_pred0.5_softmax_contras1.0_recon1.0_align100.0_pred0.5.pt",
        help="Path to pretrained model checkpoint for encoder initialization.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_name = args.model_name

    # Build wandb run name based on info / head type / attention type
    head_suffix = "multihead" if args.multihead else "singlehead"
    attn_suffix = "sparsemax" if args.sparsemax else "softmax"
    args.wandb_runname = f"{args.wandb_runname}_{args.info}_{head_suffix}_{attn_suffix}"

    start_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    log_txt = f"./results/{model_name}_{args.dataset}_{start_time}.txt"

    # ------------------------------------------------------------------
    # Dataset-specific fold and session configuration
    # ------------------------------------------------------------------
    if args.dataset == "MoBI":
        # Fixed validation/test subjects
        val_idx = [6]
        test_idx = [7]
        sessions = ["0", "1", "2"]
        args.num_eeg_channels = 59
        sbjs = list(range(8))  # 0..7
        args.n_domains = 24

        # For naming only
        val_fold = val_idx[0]
        test_fold = test_idx[0]

    elif args.dataset == "FBM":
        # Fold-based configuration
        test_fold = args.test_fold
        val_fold = (args.test_fold + 6) % 9 + 2
        val_idx = [f"0{args.test_fold}"[-2:]]
        test_idx = [f"0{args.test_fold}"[-2:]]
        sessions = ["01"]
        sbjs = ["02", "03", "04", "05", "06", "07", "08", "09", "10"]
        args.num_eeg_channels = 59
        args.motor_input_dim = 8
        args.n_domains = 9
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Overwrite wandb_runname if pretrained/scratch matters more for tagging
    args.wandb_runname = (
        f"{model_name}_{args.info}_pretrained" if args.pretrained else f"{model_name}_{args.info}_scratch"
    )

    # Construct train/val/test session lists
    train_sessions = []
    val_sessions = []
    test_sessions = []

    for idx in val_idx:
        for session in sessions:
            val_sessions.append(f"{idx}_{session}")

    for idx in test_idx:
        for session in sessions:
            test_sessions.append(f"{idx}_{session}")

    for j in sbjs:
        for session in sessions:
            train_sessions.append(f"{j}_{session}")

    # Split total batch size across GPUs on the current node
    num_gpus = max(1, torch.cuda.device_count())
    args.batch_size = max(1, args.batch_size // num_gpus)

    print("Train sessions:", train_sessions)
    print("Val sessions:  ", val_sessions)
    print("Test sessions: ", test_sessions)

    main(args)
