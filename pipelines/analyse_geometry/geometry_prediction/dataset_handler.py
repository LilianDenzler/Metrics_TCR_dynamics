# dataset_handler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder


# ----------------------------
# Configuration / data classes
# ----------------------------

@dataclass(frozen=True)
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class CVFold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class PreparedDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    df_meta: pd.DataFrame  # contains at least germline and group columns aligned to X/y


# ----------------------------
# Handler
# ----------------------------

class GeometryDatasetHandler:
    """
    Handles:
    - reading csv
    - cleaning / validating
    - converting germline to model input
    - generating splits (group-aware by tcr_name; germline holdout)
    """

    DEFAULT_TARGETS = ["BA", "BC1", "AC1", "BC2", "AC2", "dc"]
    DEFAULT_ANGLE_TARGETS = ["BA", "BC1", "AC1", "BC2", "AC2"]

    def __init__(
        self,
        germline_col: str = "germline_vj_pair",
        group_col: str = "tcr_name",
        target_cols: Optional[Sequence[str]] = None,
        angle_target_cols: Optional[Sequence[str]] = None,
        drop_na_targets: bool = True,
        collapse_rare_germlines_min_count: Optional[int] = None,
        other_label: str = "OTHER",
        random_state: int = 0,
    ) -> None:
        self.germline_col = germline_col
        self.group_col = group_col
        self.target_cols = list(target_cols) if target_cols is not None else list(self.DEFAULT_TARGETS)
        self.angle_target_cols = (
            list(angle_target_cols) if angle_target_cols is not None else list(self.DEFAULT_ANGLE_TARGETS)
        )
        self.drop_na_targets = drop_na_targets
        self.collapse_rare_germlines_min_count = collapse_rare_germlines_min_count
        self.other_label = other_label
        self.random_state = int(random_state)

        # Fit-time state for encoder
        self._encoder: Optional[OneHotEncoder] = None
        self._encoder_feature_names: Optional[List[str]] = None

    # ----------------------------
    # Loading / validation
    # ----------------------------

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load a CSV into a DataFrame."""
        df = pd.read_csv(csv_path)
        return df

    def validate_required_columns(self, df: pd.DataFrame) -> None:
        required = [self.germline_col, self.group_col] + self.target_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Present columns include: {list(df.columns)[:40]}{'...' if len(df.columns) > 40 else ''}"
            )

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning:
        - validate columns
        - ensure germline/group are strings
        - optionally drop rows with NaN targets
        - optionally collapse rare germlines into OTHER
        """
        self.validate_required_columns(df)

        out = df.copy()

        # Ensure categorical identifiers are strings (avoid mixed types)
        out[self.germline_col] = out[self.germline_col].astype(str)
        out[self.group_col] = out[self.group_col].astype(str)

        # Drop rows with NaNs in targets if requested
        if self.drop_na_targets:
            out = out.dropna(subset=self.target_cols)

        # Collapse rare germlines if requested (based on unique TCRs per germline, not raw rows)
        if self.collapse_rare_germlines_min_count is not None:
            out = self._collapse_rare_germlines(
                out,
                min_count=int(self.collapse_rare_germlines_min_count),
                label=self.other_label,
                count_unit="tcr",  # more appropriate when multiple rows per TCR might exist
            )

        # Final sanity: targets numeric
        for c in self.target_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        if self.drop_na_targets:
            out = out.dropna(subset=self.target_cols)

        return out

    def _collapse_rare_germlines(
        self,
        df: pd.DataFrame,
        min_count: int,
        label: str,
        count_unit: Literal["row", "tcr"] = "tcr",
    ) -> pd.DataFrame:
        """
        Collapse germlines with fewer than min_count into `label`.

        count_unit="tcr" counts unique group_col (tcr_name) per germline.
        """
        if min_count <= 1:
            return df

        if count_unit == "row":
            counts = df[self.germline_col].value_counts()
        else:
            # Count unique TCRs per germline
            counts = df.groupby(self.germline_col)[self.group_col].nunique()

        rare = set(counts[counts < min_count].index.tolist())
        if not rare:
            return df

        out = df.copy()
        out.loc[out[self.germline_col].isin(rare), self.germline_col] = label
        return out

    # ----------------------------
    # Preparation (X/y)
    # ----------------------------

    def fit_germline_encoder(self, df: pd.DataFrame) -> OneHotEncoder:
        """
        Fit a OneHotEncoder on germline_vj_pair.
        """
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = df[[self.germline_col]].astype(str).to_numpy()
        enc.fit(X_cat)

        # Store
        self._encoder = enc
        self._encoder_feature_names = [f"{self.germline_col}={c}" for c in enc.categories_[0].tolist()]
        return enc

    def transform_germline(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Transform germline to one-hot features. Requires fit_germline_encoder first.
        """
        if self._encoder is None or self._encoder_feature_names is None:
            raise RuntimeError("Encoder not fit. Call fit_germline_encoder(df_train) first.")

        X_cat = df[[self.germline_col]].astype(str).to_numpy()
        X = self._encoder.transform(X_cat)
        return X, list(self._encoder_feature_names)

    def extract_targets(
        self,
        df: pd.DataFrame,
        angle_encoding: Literal["raw", "sincos"] = "raw",
        degrees: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract target matrix y.

        angle_encoding:
          - "raw": return angles in degrees as provided
          - "sincos": replace each angle target with sin(angle), cos(angle) (recommended for wrap-around)
        """
        y_raw = df[self.target_cols].to_numpy(dtype=float)

        if angle_encoding == "raw":
            return y_raw, list(self.target_cols)

        if angle_encoding != "sincos":
            raise ValueError(f"Unsupported angle_encoding: {angle_encoding}")

        angle_set = set(self.angle_target_cols)
        names: List[str] = []
        cols: List[np.ndarray] = []

        for name in self.target_cols:
            col = df[name].to_numpy(dtype=float)
            if name in angle_set:
                # Convert degrees to radians for trig if needed
                rad = np.deg2rad(col) if degrees else col
                cols.append(np.sin(rad))
                cols.append(np.cos(rad))
                names.append(f"{name}_sin")
                names.append(f"{name}_cos")
            else:
                cols.append(col)
                names.append(name)

        y = np.stack(cols, axis=1)
        return y, names

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        fit_encoder: bool = False,
        angle_encoding: Literal["raw", "sincos"] = "raw",
    ) -> PreparedDataset:
        """
        Prepare X (one-hot germline) and y (geometry targets).

        Typical usage:
          - For a split: fit_encoder=True on TRAIN only, then fit_encoder=False on val/test.
        """
        df2 = self.clean_dataframe(df)

        if fit_encoder:
            self.fit_germline_encoder(df2)

        X, feat_names = self.transform_germline(df2)
        y, targ_names = self.extract_targets(df2, angle_encoding=angle_encoding)

        meta_cols = [self.germline_col, self.group_col]
        df_meta = df2[meta_cols].reset_index(drop=True)

        return PreparedDataset(
            X=X,
            y=y,
            feature_names=feat_names,
            target_names=targ_names,
            df_meta=df_meta,
        )

    # ----------------------------
    # Splitting utilities
    # ----------------------------

    def train_val_test_split_by_tcr(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        val_size: float = 0.10,
    ) -> SplitIndices:
        """
        Group-aware split:
        - test split by groups (tcr_name)
        - then val split from remaining by groups (tcr_name)

        Note: val_size is fraction of total dataset, not fraction of train.
        """
        if not (0 < test_size < 1):
            raise ValueError("test_size must be in (0,1)")
        if not (0 <= val_size < 1):
            raise ValueError("val_size must be in [0,1)")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be < 1")

        df2 = self.clean_dataframe(df)
        n = len(df2)

        groups = df2[self.group_col].to_numpy()
        idx_all = np.arange(n)

        # First split: train_val vs test
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)
        trainval_idx, test_idx = next(gss1.split(idx_all, groups=groups))

        # Second split: train vs val inside trainval
        if val_size == 0:
            return SplitIndices(train_idx=trainval_idx, val_idx=np.array([], dtype=int), test_idx=test_idx)

        # Convert val_size to fraction of trainval set
        val_frac_of_trainval = val_size / (1.0 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=self.random_state + 1)

        train_idx_rel, val_idx_rel = next(
            gss2.split(trainval_idx, groups=groups[trainval_idx])
        )
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        return SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    def group_kfold_by_tcr(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        shuffle: bool = False,
    ) -> List[CVFold]:
        """
        GroupKFold by tcr_name.

        sklearn's GroupKFold does not shuffle; if shuffle=True, we shuffle groups ourselves.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        df2 = self.clean_dataframe(df)
        n = len(df2)
        idx = np.arange(n)
        groups = df2[self.group_col].to_numpy()

        if not shuffle:
            gkf = GroupKFold(n_splits=n_splits)
            folds = []
            for i, (tr, te) in enumerate(gkf.split(idx, groups=groups)):
                folds.append(CVFold(fold=i, train_idx=tr, test_idx=te))
            return folds

        # Shuffle groups deterministically
        rng = np.random.default_rng(self.random_state)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)
        group_to_rank = {g: r for r, g in enumerate(unique_groups)}
        order = np.array([group_to_rank[g] for g in groups])
        perm = np.argsort(order, kind="stable")

        idx_perm = idx[perm]
        groups_perm = groups[perm]

        gkf = GroupKFold(n_splits=n_splits)
        folds = []
        for i, (tr_p, te_p) in enumerate(gkf.split(idx_perm, groups=groups_perm)):
            folds.append(CVFold(fold=i, train_idx=idx_perm[tr_p], test_idx=idx_perm[te_p]))
        return folds

    def group_kfold_holdout_germline(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        min_germline_tcrs: int = 2,
    ) -> List[CVFold]:
        """
        Germline-held-out CV (stress test):
        - groups are germline_vj_pair (entire germlines held out as test sets)
        - filters out germlines with too few unique TCRs (optional) to avoid degenerate folds
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        df2 = self.clean_dataframe(df)

        # Filter germlines with very few unique TCRs (to make folds feasible)
        counts = df2.groupby(self.germline_col)[self.group_col].nunique()
        keep = set(counts[counts >= int(min_germline_tcrs)].index.tolist())
        df3 = df2[df2[self.germline_col].isin(keep)].reset_index(drop=True)

        if df3.empty:
            raise ValueError("No data left after filtering germlines by min_germline_tcrs.")

        n = len(df3)
        idx = np.arange(n)
        germ_groups = df3[self.germline_col].to_numpy()

        gkf = GroupKFold(n_splits=n_splits)
        folds = []
        for i, (tr, te) in enumerate(gkf.split(idx, groups=germ_groups)):
            folds.append(CVFold(fold=i, train_idx=tr, test_idx=te))
        return folds

    # ----------------------------
    # Convenience reporting helpers
    # ----------------------------

    def germline_counts(self, df: pd.DataFrame, unit: Literal["row", "tcr"] = "tcr") -> pd.Series:
        """Return counts per germline (rows or unique TCRs)."""
        df2 = self.clean_dataframe(df)
        if unit == "row":
            return df2[self.germline_col].value_counts()
        return df2.groupby(self.germline_col)[self.group_col].nunique().sort_values(ascending=False)


# ----------------------------
# Minimal example usage
# ----------------------------
if __name__ == "__main__":
    # Example:
    # handler = GeometryDatasetHandler(collapse_rare_germlines_min_count=3, random_state=42)
    # df_unbound = handler.load_csv("unbound.csv")
    # split = handler.train_val_test_split_by_tcr(df_unbound, test_size=0.2, val_size=0.1)
    #
    # df_clean = handler.clean_dataframe(df_unbound)
    # df_train = df_clean.iloc[split.train_idx].reset_index(drop=True)
    # df_val   = df_clean.iloc[split.val_idx].reset_index(drop=True)
    # df_test  = df_clean.iloc[split.test_idx].reset_index(drop=True)
    #
    # train_ds = handler.prepare_dataset(df_train, fit_encoder=True, angle_encoding="raw")
    # val_ds   = handler.prepare_dataset(df_val,   fit_encoder=False, angle_encoding="raw")
    # test_ds  = handler.prepare_dataset(df_test,  fit_encoder=False, angle_encoding="raw")
    #
    # print(train_ds.X.shape, train_ds.y.shape)
    # print("n germlines:", len(handler._encoder_feature_names or []))
    pass
