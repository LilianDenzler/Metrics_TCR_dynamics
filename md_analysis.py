# MDAnalysis-based TCR region analysis
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from TCR_CLASS import TCR, write_pdb
# ---- your globals reused ----
CDR_FR_RANGES = {
    "A_FR1": (3,26), "A_CDR1": (27,38), "A_FR2": (39,55), "A_CDR2": (56,65),
    "A_FR3": (66,104), "A_CDR3": (105,117), "A_FR4": (118,125),
    "B_FR1": (3,26), "B_CDR1": (27,38), "B_FR2": (39,55), "B_CDR2": (56,65),
    "B_FR3": (66,104), "B_CDR3": (105,117), "B_FR4": (118,125),
}
VARIABLE_RANGE = (1,128)

#----Plotting-----
import numpy as np
import matplotlib.pyplot as plt

def _time_axis(u_traj):
    n = u_traj.trajectory.n_frames
    dt = getattr(u_traj.trajectory, "dt", None)
    if dt is None:
        return np.arange(n), "Frame"
    return np.arange(n) * float(dt), "Time (ps)"

def _nice_ylim(y, pad_frac=0.05, floor_zero=False):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0, 1)
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if ymin == ymax:
        ymin -= 1e-6
        ymax += 1e-6
    pad = (ymax - ymin) * pad_frac
    lo = ymin - pad
    hi = ymax + pad
    if floor_zero:
        lo = max(0.0, lo)
    return (lo, hi)

# ---- 2×3 grids for the six loops ----
def plot_loop_grids(loop_stats: dict, u_traj, save_prefix: str | None = None):
    order = ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"]
    t, xlabel = _time_axis(u_traj)

    # RMSD grid (per-plot scaling)
    fig_r, axes_r = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, name in zip(axes_r.ravel(), order):
        y = loop_stats[name]["rmsd"]
        ax.plot(t, y)
        ax.set_title(name)
        ax.set_ylabel("RMSD (Å)")
        ax.set_ylim(*_nice_ylim(y, pad_frac=0.07, floor_zero=True))
    for ax in axes_r[-1]:
        ax.set_xlabel(xlabel)
    fig_r.tight_layout()
    if save_prefix:
        fig_r.savefig(f"{save_prefix}_loops_rmsd.png", dpi=200)

    # TM grid (fixed 0–1)
    fig_tm, axes_tm = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, name in zip(axes_tm.ravel(), order):
        y = loop_stats[name]["tm"]
        ax.plot(t, y)
        ax.set_title(name)
        ax.set_ylabel("TM-score")
        ax.set_ylim(0.0, 1.0)
    for ax in axes_tm[-1]:
        ax.set_xlabel(xlabel)
    fig_tm.tight_layout()
    if save_prefix:
        fig_tm.savefig(f"{save_prefix}_loops_tm.png", dpi=200)

    return fig_r, fig_tm

# ---- per-bundle figures (one subplot per bundle) ----
def plot_bundle_panels(bundles: dict, u_traj, save_prefix: str | None = None,
                       keys: list[str] | None = None):
    if keys is None:
        keys = ["alpha_all_loops","beta_all_loops","all_cdrs","alpha_variable","beta_variable"]
    t, xlabel = _time_axis(u_traj)

    # RMSD (per-plot scaling)
    fig_r, axes_r = plt.subplots(len(keys), 1, figsize=(10, 2.2*len(keys)), sharex=True)
    if len(keys) == 1:
        axes_r = [axes_r]
    for ax, k in zip(axes_r, keys):
        y = bundles[k]["rmsd"]
        ax.plot(t, y)
        ax.set_ylabel("RMSD (Å)")
        ax.set_title(k)
        ax.set_ylim(*_nice_ylim(y, pad_frac=0.07, floor_zero=True))
    axes_r[-1].set_xlabel(xlabel)
    fig_r.tight_layout()
    if save_prefix:
        fig_r.savefig(f"{save_prefix}_bundles_rmsd.png", dpi=200)

    # TM (fixed 0–1)
    fig_tm, axes_tm = plt.subplots(len(keys), 1, figsize=(10, 2.2*len(keys)), sharex=True)
    if len(keys) == 1:
        axes_tm = [axes_tm]
    for ax, k in zip(axes_tm, keys):
        y = bundles[k]["tm"]
        ax.plot(t, y)
        ax.set_ylabel("TM-score")
        ax.set_title(k)
        ax.set_ylim(0.0, 1.0)
    axes_tm[-1].set_xlabel(xlabel)
    fig_tm.tight_layout()
    if save_prefix:
        fig_tm.savefig(f"{save_prefix}_bundles_tm.png", dpi=200)

    return fig_r, fig_tm

# ---------- selection helpers ----------
def _region_list(names):
    regs = []
    for n in names:
        if n in ("A_variable","B_variable"):
            regs.append((n, VARIABLE_RANGE))
        else:
            regs.append((n, CDR_FR_RANGES[n]))
    return regs

def _chain_key(chain_map, region_name):
    if region_name.startswith("A_") or region_name == "A_variable":
        return chain_map["alpha"]
    return chain_map["beta"]

def sel_for_regions(chain_map, region_names, CA_only=False):
    """
    Build an MDAnalysis selection string for (union of) IMGT regions.
    Uses segid or chainID (whichever exists).
    """
    regs = _region_list(region_names)
    parts = []
    for rname, (s,e) in regs:
        cid = _chain_key(chain_map, rname)
        atomname = " and name CA" if CA_only else ""
        # Try segid first (PDB), fall back to chainID (mmCIF)
        # We'll OR them so either will match correctly.
        parts.append(f"((segid {cid} or chainID {cid}) and resid {s}:{e}{atomname})")
    return " or ".join(parts)

# ---------- TM-score ----------
def tm_score(fixed_coords, mobile_coords):
    """
    fixed_coords, mobile_coords: (N,3) arrays with 1:1 correspondence.
    """
    d = np.linalg.norm(fixed_coords - mobile_coords, axis=1)
    N = len(d)
    if N == 0: return np.nan
    d0 = 1.24 * (max(N-15,1)**(1/3)) - 1.8
    d0 = max(d0, 0.5)
    return float(np.mean(1.0/(1.0+(d/d0)**2)))

# ---------- core: align on selection, score another selection ----------
def align_and_score(u_top_traj, u_ref, sel_align, sel_score, center=True, in_memory=True):
    """
    Align u_top_traj to u_ref using sel_align, then compute per-frame RMSD+TM on sel_score.
    Returns dict with 'rmsd', 'tm' (np.ndarray per frame), and 'n_frames'.
    """
    # 1) Align the whole trajectory in-place (on a working copy universe)
    align.AlignTraj(u_top_traj, u_ref, select=sel_align, in_memory=in_memory, center=center).run()

    # 2) Build AtomGroups for scoring (same atom order assumed across frames)
    ag_ref   = u_ref.select_atoms(sel_score)
    ag_mob   = u_top_traj.select_atoms(sel_score)

    # 3) RMSD time series (MDAnalysis)
    R = rms.RMSD(ag_mob, ag_ref, center=False, superposition=False)  # already aligned
    R.run()
    rmsd_series = R.rmsd[:,2]  # column 2 = RMSD(Å)

    # 4) TM-score time series (custom)
    tm_series = np.empty(u_top_traj.trajectory.n_frames, dtype=float)
    ref_xyz = ag_ref.positions.copy()
    for i, ts in enumerate(u_top_traj.trajectory):
        mob_xyz = ag_mob.positions.copy()
        tm_series[i] = tm_score(ref_xyz, mob_xyz)

    return {"rmsd": rmsd_series, "tm": tm_series, "n_frames": len(rmsd_series)}

# ---------- convenience wrappers for common bundles ----------
def per_loop_results(u_traj, u_ref, chain_map, CA_fit=True):
    loops = ["A_CDR1","A_CDR2","A_CDR3","B_CDR1","B_CDR2","B_CDR3"]
    out = {}
    for loop in loops:
        sel_align = sel_for_regions(chain_map, [loop], CA_only=CA_fit)
        sel_score = sel_for_regions(chain_map, [loop], CA_only=False)
        out[loop] = align_and_score(u_traj.copy(), u_ref, sel_align, sel_score)
    return out

def bundle_results(u_traj, u_ref, chain_map, CA_fit=True):
    # all alpha loops
    alpha_loops = ["A_CDR1","A_CDR2","A_CDR3"]
    beta_loops  = ["B_CDR1","B_CDR2","B_CDR3"]
    all_cdrs    = alpha_loops + beta_loops

    res = {}
    # alpha loops as one bundle
    sa = sel_for_regions(chain_map, alpha_loops, CA_only=CA_fit)
    ss = sel_for_regions(chain_map, alpha_loops, CA_only=False)
    res["alpha_all_loops"] = align_and_score(u_traj.copy(), u_ref, sa, ss)

    # beta loops as one bundle
    sb = sel_for_regions(chain_map, beta_loops, CA_only=CA_fit)
    ssb= sel_for_regions(chain_map, beta_loops, CA_only=False)
    res["beta_all_loops"] = align_and_score(u_traj.copy(), u_ref, sb, ssb)

    # all CDRs together
    sc = sel_for_regions(chain_map, all_cdrs, CA_only=CA_fit)
    ssc= sel_for_regions(chain_map, all_cdrs, CA_only=False)
    res["all_cdrs"] = align_and_score(u_traj.copy(), u_ref, sc, ssc)

    # alpha variable domain
    sva = sel_for_regions(chain_map, ["A_variable"], CA_only=CA_fit)
    ssv = sel_for_regions(chain_map, ["A_variable"], CA_only=False)
    res["alpha_variable"] = align_and_score(u_traj.copy(), u_ref, sva, ssv)

    # beta variable domain
    svb = sel_for_regions(chain_map, ["B_variable"], CA_only=CA_fit)
    ssvb= sel_for_regions(chain_map, ["B_variable"], CA_only=False)
    res["beta_variable"] = align_and_score(u_traj.copy(), u_ref, svb, ssvb)

    return res

if __name__ == "__main__":
    tcr = TCR("/mnt/larry/lilian/DATA/Cory_data/1KGC/1KGC.pdb")
    xtc_path = "/mnt/larry/lilian/DATA/Cory_data/1KGC/1KGC_Prod.xtc"
    tcr_imgt=tcr.imgt_all_structure
    write_pdb("test/tcr_imgt.pdb", tcr_imgt)
    top="test/tcr_imgt.pdb"

    u_traj = mda.Universe(top, xtc_path)
    # 2) Reference universe = frame 0 of the same topology (no traj)
    u_ref = mda.Universe(top)

    # 3) Get chain map from your TCR class (or set manually)
    # e.g., pv = tcr.pairs[0]
    chain_map = {"alpha": "A", "beta": "B"}  # or {"alpha":"G","beta":"D"} for γδ

    # 4) Per-loop (align on loop; score that loop)
    loop_stats = per_loop_results(u_traj, u_ref, chain_map, CA_fit=True)
    print("A_CDR3 RMSD:", loop_stats["A_CDR3"]["rmsd"][:5])
    print("A_CDR3 TM:  ", loop_stats["A_CDR3"]["tm"][:5])

    # 5) Bundles (alpha loops, beta loops, all CDRs, alpha/beta variable domains)
    bundles = bundle_results(u_traj, u_ref, chain_map, CA_fit=True)
    for key, res in bundles.items():
        print(key, "mean RMSD:", float(np.nanmean(res["rmsd"])), "mean TM:", float(np.nanmean(res["tm"])))
    # plots
    plot_loop_grids(loop_stats, u_traj, save_prefix="test/1KGC")
    plot_bundle_panels(bundles, u_traj, save_prefix="test/1KGC")