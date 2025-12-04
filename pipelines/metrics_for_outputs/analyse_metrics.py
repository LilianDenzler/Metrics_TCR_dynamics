import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def safe_get(d, path, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def infer_method_from_path(p: Path) -> str:
    return p.parent.name

def collect_rows(root: Path):
    files = list(root.glob("**/*metrics.json"))
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"[skip] {f}: couldn't parse JSON ({e})")
            continue
        method = infer_method_from_path(f)
        rows.append({
            "method": method,
            "file": str(f),
            "trust.gt": safe_get(data, "trust.gt"),
            "trust.pred": safe_get(data, "trust.pred"),
            "trust.combined":safe_get(data, "trust.gt")+safe_get(data, "trust.pred") if safe_get(data, "trust.gt") is not None and safe_get(data, "trust.pred") is not None else None,
            "mantel.gt.r": safe_get(data, "mantel.gt.r"),
            "mantel.gt.p": safe_get(data, "mantel.gt.p"),
            "mantel.pred.r": safe_get(data, "mantel.pred.r"),
            "mantel.pred.p": safe_get(data, "mantel.pred.p"),
            "mantel.combined.r": safe_get(data, "mantel.combined.r"),
            "mantel.combined.p": safe_get(data, "mantel.combined.p"),
        })
    return rows

def get_best_metric(root: Path, rank_by: str):
    rows = collect_rows(root)
    if not rows:
        print("No metrics files found.")
        return

    df = pd.DataFrame(rows)
    # print raw rows
    print("\n=== All metrics (per file) ===")
    print(df.sort_values(["method", "file"]).to_string(index=False))

    # summarize per method (mean across files)
    numeric_cols = [c for c in df.columns if c not in {"method","file"}]
    summary = (
        df.groupby("method")[numeric_cols]
          .agg(lambda s: np.nanmean(s.astype(float)))
          .reset_index()
    )

    # rank by chosen metric
    if rank_by not in summary.columns:
        print(f"\n[warn] rank key '{rank_by}' not found. Available numeric columns:\n  {', '.join(summary.columns.drop('method'))}")
        return

    summary_sorted = summary.sort_values(rank_by, ascending=False, na_position="last")
    best_row = summary_sorted.iloc[0]

    print("\n=== Per-method summary (mean across files) ===")
    print(summary_sorted.to_string(index=False))

    print(f"\n🏆 Best method by '{rank_by}': {best_row['method']} (score={best_row[rank_by]:.6f})")
    return best_row['method'], best_row[rank_by]


if __name__ == "__main__":
    TCR_NAME="1KGC"
    root="/workspaces/Graphormer/TCR_Metrics/outputs_dig_vanilla/1KGC"
    #rank_by="mantel.combined.r"
    rank_by="trust.combined"
    best_methods={}
    for folder in Path(root).iterdir():
        if folder.is_dir():
            if "A_variable" in str(folder):
                for subfolder in Path(root).iterdir():
                    if subfolder.is_dir():
                        region=subfolder.name
                        best_method,score=get_best_metric(subfolder, rank_by=rank_by)
                        full_region="A_variable_"+region
                        best_methods[full_region]={"method":best_method,"score":score}

            else:
                region=folder.name
                best_method,score=get_best_metric(folder, rank_by=rank_by)
                best_methods[region]={"method":best_method,"score":score}
    print("\n=== Best methods per region ===")
    best_methods_df=pd.DataFrame.from_dict(best_methods, orient="index")
    print(best_methods_df)
    best_methods_df.to_csv(f"{TCR_NAME}_best_methods_per_region_mantel.csv")



