import sys
sys.path.append("/workspaces/Graphormer/TCR_Metrics")
from TCR_TOOLS.classes.tcr import TCR
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig, LogitsOutput, ProteinType, ESMProteinError

# ------------------------
# Minimal fixes:
# 1) Load model ONCE (already done)
# 2) Use the SAME LogitsConfig that requests hidden_states
# 3) batch_embed passes sequences correctly
# 4) Filter out ESMProteinError objects (and keep df aligned)
# 5) Fix pdb_path.name crash when pdb_path is a string
# 6) Ensure string concatenation only when both seqs exist
# ------------------------

client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
EMBEDDING_CONFIG = LogitsConfig(sequence=True, return_hidden_states=True, return_embeddings=False)


def embed_sequence(sequence: str) -> LogitsOutput:
    protein = ESMProtein(sequence=str(sequence))
    protein_tensor = client.encode(protein)
    # FIX: request hidden states (you later use output.hidden_states)
    logits_output = client.logits(protein_tensor, EMBEDDING_CONFIG)
    return logits_output


def batch_embed(inputs: Sequence[ProteinType]) -> Sequence[LogitsOutput]:
    """
    Minimal change: your inputs are sequences (strings), so call embed_sequence(model, seq).
    NOTE: Threading + a single GPU model can cause instability; keeping it because you asked minimal changes.
    If you see many failures, set max_workers=1 or remove ThreadPoolExecutor.
    """
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(embed_sequence, seq) for seq in inputs]
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append(ESMProteinError(500, str(e)))
    return results


def plot_embeddings_at_layer(
    all_mean_embeddings: Sequence[torch.Tensor],
    df_kept: pd.DataFrame,
    layer_idx: int,
    N_KMEANS_CLUSTERS: int,
    seq_col_name: str,
    classes_col="beta_vj"
):
    stacked_mean_embeddings = (
        torch.stack([embedding[layer_idx, :] for embedding in all_mean_embeddings])
        .float()
        .numpy()
    )

    pca = PCA(n_components=2)
    projected_mean_embeddings = pca.fit_transform(stacked_mean_embeddings)

    kmeans = KMeans(n_clusters=N_KMEANS_CLUSTERS, random_state=0, n_init="auto").fit(projected_mean_embeddings)
    rand_index = adjusted_rand_score(df_kept[classes_col].astype(str), kmeans.labels_)

    plt.figure(figsize=(24, 24))
    sns.scatterplot(
        x=projected_mean_embeddings[:, 0],
        y=projected_mean_embeddings[:, 1],
        hue=df_kept[classes_col].astype(str),
        s=30,
        linewidth=0,
    )
    plt.title(f"PCA of mean embeddings at layer {layer_idx}.\nRand index: {rand_index:.2f}")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(
        f"/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/geometry_prediction/plots/pca_{str(layer_idx)}_{seq_col_name}.png"
    )
    plt.close()


def run_layer_plotter(df: pd.DataFrame, N_KMEANS_CLUSTERS: int, seq_col_name="alpha_variable_seq", classes_col="alpha_vj"):
    # Embed alpha sequences
    seqs = df[seq_col_name].tolist()
    outputs = batch_embed(seqs)

    # FIX: filter out failures and keep df aligned
    kept_embeddings = []
    kept_rows = []

    for i, out in enumerate(outputs):
        if isinstance(out, ESMProteinError):
            continue
        if not hasattr(out, "hidden_states") or out.hidden_states is None:
            continue

        hs = out.hidden_states
        hs_t = hs if isinstance(hs, torch.Tensor) else torch.as_tensor(hs)
        hs_t = hs_t.float()

        # your original pooling logic (mean over sequence dimension)
        # hidden_states expected shape [num_layers, seq_len, hidden] or similar
        # dim=-2 corresponds to seq_len for [layers, seq, hidden]
        mean_emb = torch.mean(hs_t, dim=-2).squeeze()  # -> [num_layers, hidden]
        kept_embeddings.append(mean_emb.detach().cpu())
        kept_rows.append(i)

    print(f"[INFO] Embedded: {len(outputs)} | kept: {len(kept_embeddings)} | dropped: {len(outputs)-len(kept_embeddings)}")

    if len(kept_embeddings) == 0:
        raise RuntimeError("No valid embeddings found. Check sequences and embedding config.")

    print("embedding shape [num_layers, hidden_size]:", kept_embeddings[0].shape)

    df_kept = df.iloc[kept_rows].reset_index(drop=True)

    # Avoid layer index out of bounds
    num_layers = kept_embeddings[0].shape[0]
    for layer_idx in range(0,30):
        if layer_idx < num_layers:
            plot_embeddings_at_layer(kept_embeddings, df_kept, layer_idx=layer_idx, N_KMEANS_CLUSTERS=N_KMEANS_CLUSTERS, seq_col_name=seq_col_name,classes_col=classes_col)


if __name__ == "__main__":
    out_path = "/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data/TCR3d_unpaired_unbound_with_variable_seqs.csv"
    unbound_upaired_data_path = "/workspaces/Graphormer/TCR_Metrics/pipelines/analyse_geometry/data/TCR3d_unpaired_unbound.csv"

    df = pd.read_csv(unbound_upaired_data_path)

    # If cached file exists, use it
    if os.path.isfile(out_path):
        df = pd.read_csv(out_path)
    else:
        # Ensure these columns exist
        for col in ["alpha_variable_seq", "beta_variable_seq", "ab_variable_seq"]:
            if col not in df.columns:
                df[col] = pd.NA

        for idx, row in df.iterrows():
            pdb_path = row["pdb_path"]
            try:
                tcr = TCR(
                    input_pdb=str(pdb_path),
                    traj_path=None,
                    contact_cutoff=5.0,
                    min_contacts=50,
                    legacy_anarci=True,
                )
                pair = tcr.pairs[0]
                seqs = pair.cdr_fr_sequences()
                A_var_seq = seqs.get("A_variable", None)
                B_var_seq = seqs.get("B_variable", None)

                df.at[idx, "alpha_variable_seq"] = A_var_seq
                df.at[idx, "beta_variable_seq"] = B_var_seq

                if isinstance(A_var_seq, str) and isinstance(B_var_seq, str):
                    df.at[idx, "ab_variable_seq"] = A_var_seq + B_var_seq
                else:
                    df.at[idx, "ab_variable_seq"] = pd.NA

            except Exception as e:
                # FIX: pdb_path might be a string; .name would crash
                print(f"  [WARN] Failed to initialise TCR for {pdb_path}: {e}")
                continue

        df.to_csv(out_path, index=False)

    number_germline_classes = df["germline_vj_pair"].astype(str).nunique()
    print("NUMBER OF GERMLINES", number_germline_classes)

    # Drop rows missing germline or seq
    df = df.dropna(subset=["germline_vj_pair", "alpha_variable_seq","beta_variable_seq","alpha_vj","beta_vj"]).reset_index(drop=True)

    run_layer_plotter(df, number_germline_classes, seq_col_name="alpha_variable_seq",classes_col="alpha_vj")
    run_layer_plotter(df, number_germline_classes, seq_col_name="beta_variable_seq", classes_col="beta_vj")
