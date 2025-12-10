import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from types import SimpleNamespace

from models.contraDCN import EEG_pred_multihead
from utils.myaml import load_config
from utils.utils import prepare_GPP_dataset_vis

# ---------------------------------------------------------------------
# Matplotlib configuration (Times-like fonts, suitable for papers)
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})
plt.rcParams["font.size"] = 18


def visualize_tsne_with_pca(embeddings, phases, save_path="tsne_pca_eeg_embeddings.png"):
    """
    Run PCA + t-SNE on embeddings and plot a 2D t-SNE visualization, colored by phase.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(phases, torch.Tensor):
        phases = phases.detach().cpu().numpy()

    phases = np.squeeze(phases)

    valid_mask = phases != -1
    embeddings = embeddings[valid_mask]
    phases = phases[valid_mask]

    if embeddings.shape[0] < 5:
        raise ValueError("Not enough samples for t-SNE visualization.")

    n_pca = min(10, embeddings.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    perplexity = min(30, max(5, embeddings_pca.shape[0] // 10))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    embeddings_tsne = tsne.fit_transform(embeddings_pca)

    phase_colors = {0: "red", 1: "blue", 2: "orange", 3: "green"}
    phase_names = {0: "1", 1: "2", 2: "3", 3: "4"}

    plt.figure(figsize=(8, 6))
    for p in sorted(phase_colors.keys()):
        mask = phases == p
        if np.any(mask):
            plt.scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=phase_colors[p],
                label=phase_names[p],
                alpha=0.7,
                s=10,
            )

    plt.title("t-SNE (after PCA) of EEG Embeddings")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_tsne_compare(
    before_embed,
    before_phase,
    after_embed,
    after_phase,
    save_path="tsne_compare.png",
):
    """
    Plot a two-row t-SNE comparison:
    - Row 1: embeddings before Stage I (random initialization)
    - Row 2: embeddings after Stage I (trained / loaded weights)
    """

    def run_tsne(embeddings, phases):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if isinstance(phases, torch.Tensor):
            phases = phases.detach().cpu().numpy()

        phases = np.squeeze(phases)
        valid_mask = phases != -1
        embeddings = embeddings[valid_mask]
        phases = phases[valid_mask]

        if embeddings.shape[0] < 5:
            raise ValueError("Not enough samples for t-SNE visualization.")

        n_pca = min(10, embeddings.shape[1])
        pca = PCA(n_components=n_pca, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)

        perplexity = min(30, max(5, embeddings_pca.shape[0] // 10))
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        embeddings_tsne = tsne.fit_transform(embeddings_pca)
        return embeddings_tsne, phases

    phase_colors = {0: "red", 1: "blue", 2: "orange", 3: "green"}
    phase_names = {0: "1", 1: "2", 2: "3", 3: "4"}

    tsne_before, phase_before = run_tsne(before_embed, before_phase)
    tsne_after, phase_after = run_tsne(after_embed, after_phase)

    fig, axs = plt.subplots(2, 1, figsize=(8, 12), sharex=True, sharey=True)

    # Before Stage I
    for p in phase_colors:
        mask = phase_before == p
        if np.any(mask):
            axs[0].scatter(
                tsne_before[mask, 0],
                tsne_before[mask, 1],
                c=phase_colors[p],
                s=10,
                alpha=0.7,
                label=phase_names[p],
            )
    axs[0].set_title("Before Stage I", fontsize=18)
    axs[0].grid(True)
    axs[0].legend(fontsize=12)

    # After Stage I
    for p in phase_colors:
        mask = phase_after == p
        if np.any(mask):
            axs[1].scatter(
                tsne_after[mask, 0],
                tsne_after[mask, 1],
                c=phase_colors[p],
                s=10,
                alpha=0.7,
                label=phase_names[p],
            )
    axs[1].set_title("After Stage I", fontsize=18)
    axs[1].grid(True)
    axs[1].legend(fontsize=12)

    fig.suptitle(
        "t-SNE of EEG Embeddings from NeuroDyGait",
        fontsize=24,
        fontweight="bold",
        y=0.98,
    )

    axs[1].set_xlabel("t-SNE Dim 1", fontsize=16)
    axs[0].set_ylabel("t-SNE Dim 2", fontsize=16)
    axs[1].set_ylabel("t-SNE Dim 2", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def extract_embeddings(model, dataloader, device):
    """
    Run a model over all batches and collect latent embeddings and phase labels.
    Assumes dataloader yields dicts with keys: 'EEG_X', 'phase', and optionally 'domain', 'Y', 'transitions'.
    """
    all_embeds = []
    all_phases = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            eeg_x = batch["EEG_X"]
            if isinstance(eeg_x, np.ndarray):
                eeg_x = np.expand_dims(eeg_x, axis=1)
                batch["EEG_X"] = torch.from_numpy(eeg_x)
            else:
                batch["EEG_X"] = eeg_x.unsqueeze(1)

            phase = batch["phase"]
            if isinstance(phase, torch.Tensor):
                mask_torch = phase != -1
                mask_np = mask_torch.cpu().numpy()
            else:
                mask_np = phase != -1
                mask_torch = torch.from_numpy(mask_np)

            for key, value in batch.items():
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    batch[key] = value[mask_torch]
                elif isinstance(value, np.ndarray):
                    batch[key] = value[mask_np]

            data = {
                "EEG": batch["EEG_X"].float().to(device),
                "phase": batch["phase"].to(device),
                "domain": batch.get("domain", None),
                "Y": batch.get("Y", None),
                "transitions": batch.get("transitions", None),
            }

            embeds = model(data, vis_embd=True)
            all_embeds.append(embeds.detach().cpu().numpy())
            all_phases.append(data["phase"].detach().cpu().numpy())

    all_embeds = np.concatenate(all_embeds, axis=0)
    all_phases = np.concatenate(all_phases, axis=0)
    return all_embeds, all_phases


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 1. Load pretrained model checkpoint
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_path = "./results/GPP_ind/ContraDCN_pred/ckpt_fold9_0.6727.pt"

    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    args_dict = checkpoint["args"]
    args = SimpleNamespace(**args_dict)

    # -----------------------------------------------------------------
    # 2. Build session splits and dataset for phase visualization
    # -----------------------------------------------------------------
    mode = "normal"
    if mode == "normal":
        val_idx = [i for i in range(40, 45)]
        test_idx = [i for i in range(45, 50)]
        sbjs = [i for i in range(50)]
    else:
        val_idx = [1]
        test_idx = [1]
        sbjs = [1]

    sessions = ["2", "1"]
    args.num_eeg_channels = 59
    args.dataset_dir = "/home/fuxi/FT_dataset_PhaseTrans/"
    args.n_domains = 100

    train_sessions = []
    val_sessions = []
    test_sessions = []

    for idx in val_idx:
        for s in sessions:
            val_sessions.append(f"{idx}_{s}")
    for idx in test_idx:
        for s in sessions:
            test_sessions.append(f"{idx}_{s}")
    for j in sbjs:
        for s in sessions:
            train_sessions.append(f"{j}_{s}")

    val_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in val_sessions]
    test_domains = [int(s.split("_")[0]) * 2 + int(s.split("_")[1]) - 3 for s in test_sessions]
    target_domains = val_domains + test_domains

    dataset_train, dataset_val, dataset_test = prepare_GPP_dataset_vis(
        args.dataset_dir, train_sessions, val_sessions, test_sessions
    )
    print("Finished loading dataset.")
    print("Train dataset size:", len(dataset_train))

    # For visualization we only need the training split
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=128, shuffle=True, num_workers=0
    )

    # -----------------------------------------------------------------
    # 3. Build models: before and after Stage I
    # -----------------------------------------------------------------
    print(">>> Extracting BEFORE Stage I embeddings (random initialization)...")
    model_before = EEG_pred_multihead(
        num_eeg_channels=args.num_eeg_channels,
        time_steps=args.time_steps,
        embed_dim=args.embed_dim,
        kernel_width=args.kernel_width,
        pool_width=args.pool_width,
        nFiltLaterLayer=args.nFiltLaterLayer,
        target_domains=target_domains,
    ).float().to(device)

    before_embed, before_phase = extract_embeddings(model_before, data_loader_train, device)

    print(">>> Extracting AFTER Stage I embeddings (loaded weights)...")
    model_after = EEG_pred_multihead(
        num_eeg_channels=args.num_eeg_channels,
        time_steps=args.time_steps,
        embed_dim=args.embed_dim,
        kernel_width=args.kernel_width,
        pool_width=args.pool_width,
        nFiltLaterLayer=args.nFiltLaterLayer,
        target_domains=target_domains,
    ).float().to(device)
    model_after.load_state_dict(state_dict)

    after_embed, after_phase = extract_embeddings(model_after, data_loader_train, device)

    # -----------------------------------------------------------------
    # 4. Visualizations
    # -----------------------------------------------------------------
    visualize_tsne_with_pca(after_embed, after_phase, save_path="tsne_pca_after_stageI.png")

    visualize_tsne_compare(
        before_embed,
        before_phase,
        after_embed,
        after_phase,
        save_path="tsne_compare.png",
    )

    print("Saved: tsne_pca_after_stageI.png")
    print("Saved: tsne_compare.png")
