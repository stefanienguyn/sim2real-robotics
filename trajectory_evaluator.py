"""Robot locomotion accuracy analysis against a Motive (OptiTrack) CSV.

Note: the reference track is an open A->B path, NOT a closed loop, and is never
re-indexed. Start alignment trims the ROBOT's lead-in so its first frame lands
on the track's start; the track is left intact, so no end-to-start closing
segment (loop) is ever drawn.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial import KDTree

# Motive CSV fixed row indices (0-based)
_ROW_TYPE = 2
_ROW_NAME = 3
_ROW_SUBHEADER = 6
_ROW_AXIS = 7
_ROW_DATA = 8

COL_INDICES = [0, 1, 6, 8]  # fallback for single-body CSVs
COL_NAMES = ["frame", "time", "x", "z"]

NOISE_STD = 5.0
DRIFT_MAX = 15.0
TIME_SCALE = 1.15
FRAME_SCALE = 1.30
SMOOTH_WIN = 15

# mm; a frame-to-frame move larger than this (at 120 Hz) is a tracking teleport
# / dropout bridge, not real robot motion, and gets cut out.
MAX_STEP_MM = 500.0

# Trailing-idle trim: if near the end of a take the robot settles inside a spot
# of IDLE_RADIUS_MM for at least IDLE_MIN_S, that dwell (and any brief end-of-run
# handling after it) isn't track-following and is dropped.
IDLE_RADIUS_MM = 20.0
IDLE_MIN_S = 10.0
IDLE_TAIL_GRACE_S = 30.0

# mm; skip start alignment if the nearest track point is farther than this
# from the robot's start (indicates the two takes don't share a start region).
START_MATCH_TOL = 200.0

# Dark theme palette
_BG = "#0F1923"
_PANEL = "#16232E"
_GRID = "#1E3040"
_TEXT = "#E0E8F0"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def _read_header_row(path: Path, row_index: int) -> list:
    df = pd.read_csv(
        path, header=None, skiprows=row_index, nrows=1, dtype=str, encoding="latin-1"
    )
    return df.iloc[0].fillna("").tolist()


def _find_body_cols(path: Path, body_name: str) -> list:
    """Locate [frame_col, time_col, x_col, z_col] for the named rigid body.

    Reads Type, Name, Subheader, and Axis label rows explicitly so that X/Z
    are matched by their label rather than by positional offset.
    """
    types = _read_header_row(path, _ROW_TYPE)
    names = _read_header_row(path, _ROW_NAME)
    subhdrs = _read_header_row(path, _ROW_SUBHEADER)
    axes = _read_header_row(path, _ROW_AXIS)

    x_col = z_col = None
    for i, (t, n, s, a) in enumerate(zip(types, names, subhdrs, axes)):
        if (
            t.strip() == "Rigid Body"
            and n.strip() == body_name
            and s.strip() == "Position"
        ):
            label = a.strip().upper()
            if label == "X":
                x_col = i
            elif label == "Z":
                z_col = i

    if x_col is None or z_col is None:
        available = sorted(
            {
                n.strip()
                for t, n in zip(types, names)
                if t.strip() == "Rigid Body" and n.strip()
            }
        )
        raise ValueError(
            f"Rigid body '{body_name}' Position X/Z not found in {path}.\n"
            f"Available rigid bodies: {available}\n"
            "Use --robot_body / --track_body to specify the correct name."
        )
    return [0, 1, x_col, z_col]


def load_motive_csv(
    path: Path, body_name: Optional[str] = None, scale: float = 1.0
) -> pd.DataFrame:
    """Load frame/time/x/z for a named rigid body from a Motive CSV.

    body_name: rigid body label in the Name row. When None, falls back to the
               hardcoded COL_INDICES (suitable for single-body track.csv).
    scale:     multiplied onto x/z after loading (e.g. 0.001 converts mm to m).
    """
    col_indices = _find_body_cols(path, body_name) if body_name else COL_INDICES
    df = pd.read_csv(path, skiprows=_ROW_DATA, header=0, encoding="latin-1")
    df = df.iloc[:, col_indices].copy()
    df.columns = COL_NAMES
    before = len(df)
    df = df.dropna().astype(float).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  ({dropped} dropout frames removed)")
    if scale != 1.0:
        df["x"] *= scale
        df["z"] *= scale
    return df


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------


def generate_mock_robot(
    track: pd.DataFrame, out_path: Path, rng: np.random.Generator
) -> pd.DataFrame:
    t0, t1 = track["time"].iloc[0], track["time"].iloc[-1]
    n = int(len(track) * FRAME_SCALE)
    robot_time = np.linspace(t0, t1 * TIME_SCALE, n)
    sample_time = np.linspace(t0, t1, n)
    track_t = track["time"].to_numpy()
    spline_x = CubicSpline(track_t, track["x"].to_numpy())
    spline_z = CubicSpline(track_t, track["z"].to_numpy())
    xs = (
        spline_x(sample_time)
        + rng.normal(0, NOISE_STD, n)
        + np.linspace(0, DRIFT_MAX, n)
    )
    zs = spline_z(sample_time) + rng.normal(0, NOISE_STD, n)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        for i in range(_ROW_DATA):
            f.write(f"# Mock robot recording - metadata row {i + 1}\n")
        f.write("Frame,Time (Seconds),,,,,X,,Z\n")
        for frame, ts, x, z in zip(np.arange(1, n + 1), robot_time, xs, zs):
            f.write(f"{int(frame)},{ts:.6f},0,0,0,0,{x:.4f},0,{z:.4f}\n")
    return pd.DataFrame(
        {
            "frame": np.arange(1, n + 1, dtype=float),
            "time": robot_time,
            "x": xs,
            "z": zs,
        }
    )


# ---------------------------------------------------------------------------
# Trajectory cleaning
# ---------------------------------------------------------------------------


def _jump_break_indices(xz: np.ndarray, max_step: float) -> np.ndarray:
    """Indices i where the step from frame i to i+1 exceeds ``max_step`` mm.

    A rigid body can't move that far between 120 Hz frames, so such a step is a
    tracking teleport / dropout bridge rather than real motion. These are the
    joints the path plot must NOT connect with a straight line, and the markers
    the tail trim uses to find where an end-of-run excursion begins.
    """
    steps = np.hypot(np.diff(xz[:, 0]), np.diff(xz[:, 1]))
    return np.where(steps > max_step)[0]  # step k spans frame k -> k+1


def _trim_trailing_run(
    robot: pd.DataFrame,
    *,
    max_step: float,
    radius: float,
    min_seconds: float,
    tail_grace: float = IDLE_TAIL_GRACE_S,
) -> pd.DataFrame:
    """Trim end-of-run junk while keeping the whole track-following run.

    At the end of a take the robot stops following the track: it parks, or a
    tracking teleport throws it off and it never recovers. We find the terminal
    park — the longest stretch where the robot stays within ``radius`` mm over a
    ``min_seconds`` window and that sits at the very end of the take — and cut
    there. If a teleport lands within ``tail_grace`` seconds before that park,
    we cut back to the teleport so the off-track lurch goes too.

    Crucially, mid-run teleports that the robot recovers from are LEFT IN PLACE
    (the plot simply doesn't draw the bridge). A dropout partway through a run no
    longer discards everything after it — that was the old bug.
    """
    xz = robot[["x", "z"]].to_numpy()
    tm = robot["time"].to_numpy()
    n = len(robot)
    if n < 3 or (tm[-1] - tm[0]) < min_seconds:
        return robot

    # --- terminal-park onset -------------------------------------------------
    win_end = np.clip(np.searchsorted(tm, tm + min_seconds, side="right") - 1, 0, n - 1)
    idx = np.arange(n)
    # net displacement over the window (not path length: mocap jitter inflates
    # path length even when the robot is stationary).
    disp = np.hypot(xz[win_end, 0] - xz[:, 0], xz[win_end, 1] - xz[:, 1])
    parked = (win_end > idx) & (disp < radius)

    cut = n
    if parked.any():
        best_s, best_e, i = 0, 0, 0
        while i < n:
            if parked[i]:
                j = i
                while j < n and parked[j]:
                    j += 1
                if j - i > best_e - best_s:
                    best_s, best_e = i, j
                i = j
            else:
                i += 1
        dwell_end_t = tm[int(win_end[best_e - 1])]
        if (
            dwell_end_t - tm[best_s] >= min_seconds
            and tm[-1] - dwell_end_t <= tail_grace  # else driving resumes -> mid-run pause
        ):
            cut = best_s

    # --- if a teleport leads straight into that junk, cut back to it ----------
    breaks = _jump_break_indices(xz, max_step)
    if cut < n and len(breaks):
        lead_in = breaks[(breaks < cut) & (tm[cut] - tm[breaks] <= tail_grace)]
        if len(lead_in):
            cut = int(lead_in.min()) + 1

    if cut >= n:
        return robot
    print(
        f"  Tail trim: cut at t={tm[cut]:.0f}s; dropped {n - cut} trailing "
        "frames (off-track / idle end-of-run)"
    )
    return robot.iloc[:cut].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Deviation analysis
# ---------------------------------------------------------------------------


def compute_deviation(track: pd.DataFrame, robot: pd.DataFrame) -> np.ndarray:
    tree = KDTree(track[["x", "z"]].to_numpy())
    distances, _ = tree.query(robot[["x", "z"]].to_numpy())
    return distances


# ---------------------------------------------------------------------------
# Coordinate alignment
# ---------------------------------------------------------------------------


def align_robot_to_track(track: pd.DataFrame, robot: pd.DataFrame) -> pd.DataFrame:
    """Best-fit the robot onto the track, but never worse than leaving it as-is.

    OptiTrack takes recorded in the same calibrated room already share a global
    frame, so the identity (no-op) transform is usually optimal and any forced
    transform only hurts. This is easy to get wrong because the two takes can
    cover different extents (different durations / dwell regions), which pulls
    their centroids apart — so a centroid-seeded search happily slides an
    already-aligned robot into a wrong local minimum.

    Guardrails:
      * rotate about the robot centroid, not the origin, so a small angle isn't
        amplified into a large lever-arm translation (better conditioning);
      * search from two starts — identity and the centroid offset;
      * keep a transform ONLY if it beats the identity baseline. Otherwise the
        robot is returned unchanged.
    """
    t = track[["x", "z"]].to_numpy()
    r = robot[["x", "z"]].to_numpy()
    tree = KDTree(t)
    rc = r.mean(axis=0)

    def _mae(params):
        dx, dz, deg = params
        rad = np.radians(deg)
        R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        moved = (R @ (r - rc).T).T + rc + [dx, dz]
        return tree.query(moved)[0].mean()

    baseline = _mae([0.0, 0.0, 0.0])
    dx0, dz0 = t.mean(axis=0) - rc
    best_params, best_mae = [0.0, 0.0, 0.0], baseline
    for start in ([0.0, 0.0, 0.0], [dx0, dz0, 0.0]):
        res = minimize(
            _mae,
            start,
            method="Nelder-Mead",
            options={"xatol": 0.5, "fatol": 0.01, "maxiter": 2000},
        )
        if res.fun < best_mae:
            best_params, best_mae = list(res.x), float(res.fun)

    if best_mae >= baseline - 1e-6:
        print(
            f"Alignment: none applied — takes already share a frame "
            f"(baseline MAE {baseline:.1f} mm)"
        )
        return robot

    dx, dz, deg = best_params
    rad = np.radians(deg)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    r_aligned = (R @ (r - rc).T).T + rc + [dx, dz]
    print(
        f"Alignment: translation ({dx:+.1f}, {dz:+.1f}) mm, rotation {deg:+.2f}° "
        f"(MAE {baseline:.1f} -> {best_mae:.1f} mm)"
    )
    aligned = robot.copy()
    aligned["x"] = r_aligned[:, 0]
    aligned["z"] = r_aligned[:, 1]
    return aligned


def align_start_anchor(track: pd.DataFrame, robot: pd.DataFrame) -> pd.DataFrame:
    """Translate the whole robot so its first frame sits exactly on the track's
    first frame (opt-in alternative to ``align_robot_to_track``).

    Simple, deterministic, and makes the start markers coincide exactly. The
    trade-off: it pins one point, so any difference between the two takes' start
    positions becomes a systematic offset applied to the entire path — usually a
    few mm worse MAE than the whole-path best fit. Use when an exact start match
    matters more than the lowest possible error.
    """
    shift = (
        track[["x", "z"]].iloc[0].to_numpy() - robot[["x", "z"]].iloc[0].to_numpy()
    )
    print(
        f"Start-anchor: shifted robot by ({shift[0]:+.1f}, {shift[1]:+.1f}) mm "
        "so its start coincides with the track's start"
    )
    anchored = robot.copy()
    anchored["x"] = robot["x"] + shift[0]
    anchored["z"] = robot["z"] + shift[1]
    return anchored


def _align_robot_start(track: pd.DataFrame, robot: pd.DataFrame) -> pd.DataFrame:
    """Trim the robot's lead-in so it begins at the track's start.

    The track (ground truth) is the anchor and is never modified — it stays an
    open A->B path, so no end-to-start closing segment is ever drawn. We take
    the track's first point, find the robot frame nearest to it (searching only
    the robot's early portion, so a later pass near the start line can't be
    picked) and drop every robot frame before it. The robot's start then
    coincides with the track's start. Only the robot is trimmed, so the track's
    deviation reference set is unchanged; the dropped lead-in frames simply
    aren't scored.
    """
    track_start = track[["x", "z"]].iloc[0].to_numpy()
    search_n = min(len(robot), max(200, len(robot) // 5))
    robot_sub = robot[["x", "z"]].iloc[:search_n].to_numpy()
    dist, idx = KDTree(robot_sub).query(track_start)
    if dist > START_MATCH_TOL:
        print(
            f"  Start align skipped (nearest robot frame is {dist:.1f} mm "
            "from track start)"
        )
        return robot
    if idx == 0:
        return robot
    trimmed = robot.iloc[idx:].reset_index(drop=True)
    print(
        f"  Start align: dropped {idx} leading robot frames; robot now starts "
        f"{dist:.1f} mm from track start"
    )
    return trimmed


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(errors: np.ndarray) -> dict:
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "std": float(np.std(errors)),
    }


def print_summary(nn_errors: np.ndarray) -> None:
    nn = compute_metrics(nn_errors)
    sep = "=" * 52
    print(f"\n{sep}")
    print("  TRAJECTORY EVALUATION REPORT")
    print(sep)
    print("  -- Nearest-Neighbour Spatial Matching --")
    print(f"     MAE        : {nn['mae']:.2f} mm")
    print(f"     RMSE       : {nn['rmse']:.2f} mm")
    print(f"     Max Error  : {nn['max_error']:.2f} mm")
    print(f"     Std Dev    : {nn['std']:.2f} mm")
    print(f"     % <= 10 mm : {100 * (nn_errors < 10).mean():.1f}%")
    print(f"     % <= 20 mm : {100 * (nn_errors < 20).mean():.1f}%")
    print(f"     % <= 50 mm : {100 * (nn_errors < 50).mean():.1f}%")
    print(sep)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _apply_dark_axes(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(color=_GRID, linewidth=0.5)


def plot_all(
    track: pd.DataFrame,
    robot: pd.DataFrame,
    nn_errors: np.ndarray,
    out_path: Path,
    dark: bool = True,
    jump_times: Optional[np.ndarray] = None,
) -> None:
    if dark:
        sns.set_theme(style="darkgrid")
        fig_fc = _BG
        track_c = "#4FC3F7"
        robot_c = "#FF7043"
        cmap = "plasma"
        txt_c = _TEXT
    else:
        sns.reset_defaults()
        fig_fc = "white"
        track_c = "steelblue"
        robot_c = "darkorange"
        cmap = "RdYlGn_r"
        txt_c = "black"

    fig = plt.figure(figsize=(18, 10), facecolor=fig_fc)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    if dark:
        for ax in (ax1, ax2, ax3):
            _apply_dark_axes(ax)

    fig.suptitle(
        "Robot Locomotion Accuracy Analysis",
        fontsize=14,
        fontweight="bold",
        color=txt_c,
    )

    lkw = dict(
        fontsize=7,
        facecolor=fig_fc,
        labelcolor=txt_c,
        edgecolor=_GRID if dark else "gray",
        loc="best",
    )

    # Shared axis limits for Panels 1 & 2 (cover both track and robot)
    all_x = np.concatenate([track["x"].to_numpy(), robot["x"].to_numpy()])
    all_z = np.concatenate([track["z"].to_numpy(), robot["z"].to_numpy()])
    _pad = max((all_x.max() - all_x.min()) * 0.05, 1.0)
    _xlim = (all_x.min() - _pad, all_x.max() + _pad)
    _zlim = (all_z.min() - _pad, all_z.max() + _pad)

    # Teleport joints (detected on the full-res robot and passed in as times, so
    # striding can't be mistaken for a teleport): don't connect them with a
    # straight line. Map each jump time to the strided segment that spans it.
    r_xz = robot[["x", "z"]].to_numpy()
    tm = robot["time"].to_numpy()
    if jump_times is None or len(jump_times) == 0:
        breaks = np.empty(0, dtype=int)
    else:
        jt = np.asarray(jump_times, dtype=float)
        jt = jt[(jt > tm[0]) & (jt < tm[-1])]
        breaks = np.unique(np.clip(np.searchsorted(tm, jt) - 1, 0, len(r_xz) - 2))

    # -- Panel 1: path overlay -----------------------------------------------
    ax1.plot(
        track["x"], track["z"], color=track_c, lw=1.5, label="Track (ground truth)"
    )
    rx = np.insert(robot["x"].to_numpy(), breaks + 1, np.nan)  # NaN breaks the line
    rz = np.insert(robot["z"].to_numpy(), breaks + 1, np.nan)
    ax1.plot(rx, rz, color=robot_c, lw=1.0, alpha=0.85, label="Robot path")
    for df, mk in [(track, "o"), (robot, "^")]:
        ax1.scatter(
            df["x"].iloc[0], df["z"].iloc[0], color="lime", s=70, marker=mk, zorder=5
        )
        ax1.scatter(
            df["x"].iloc[-1],
            df["z"].iloc[-1],
            color="yellow",
            s=70,
            marker="X",
            zorder=5,
        )
    ax1.set_xlim(_xlim)
    ax1.set_ylim(_zlim)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Z (mm)")
    ax1.set_title("Path Overlay")
    ax1.legend(**lkw)

    # -- Panel 2: LineCollection deviation heatmap ---------------------------
    r_pts = r_xz
    seg_keep = np.ones(len(r_pts) - 1, dtype=bool)
    seg_keep[breaks] = False  # drop teleport segments so no bridge is coloured in
    segs = list(
        np.concatenate([r_pts[:-1, None], r_pts[1:, None]], axis=1)[seg_keep]
    )
    seg_err = ((nn_errors[:-1] + nn_errors[1:]) / 2)[seg_keep]
    norm = mcolors.Normalize(vmin=nn_errors.min(), vmax=nn_errors.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm, lw=2.5, alpha=0.95)
    lc.set_array(seg_err)
    ax2.add_collection(lc)
    ax2.plot(
        track["x"],
        track["z"],
        color=track_c,
        lw=1,
        alpha=0.3,
        ls="--",
        zorder=1,
        label="Track",
    )
    cb = fig.colorbar(lc, ax=ax2, pad=0.02, fraction=0.046)
    cb.set_label("Deviation (mm)", color=txt_c, fontsize=8)
    if dark:
        cb.ax.yaxis.set_tick_params(color=_TEXT, labelcolor=_TEXT, labelsize=7)
        cb.outline.set_edgecolor(_GRID)  # type: ignore[union-attr]
    ax2.set_xlim(_xlim)
    ax2.set_ylim(_zlim)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Z (mm)")
    ax2.set_title("Deviation Heatmap")
    ax2.legend(**lkw)

    # -- Panel 3: deviation over real time -----------------------------------
    raw_dev = pd.Series(nn_errors)
    smooth_dev = raw_dev.rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    t_axis = robot["time"]
    ax3.fill_between(t_axis, raw_dev, alpha=0.15, color=robot_c)
    ax3.plot(t_axis, raw_dev, color=robot_c, lw=0.8, label="NN deviation (raw)")
    ax3.plot(
        t_axis,
        smooth_dev,
        color=robot_c,
        lw=1.8,
        label=f"NN deviation (rolling mean, w={SMOOTH_WIN})",
    )
    ax3.axhline(10, color="gold", lw=1.2, ls=":", label="10 mm")
    ax3.axhline(20, color="tomato", lw=1.2, ls=":", label="20 mm")
    ax3.axhline(50, color="firebrick", lw=1.2, ls=":", label="50 mm")
    ax3.set_xlim(t_axis.min(), t_axis.max())
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Deviation (mm)")
    ax3.set_title("Deviation Over Time")
    ax3.legend(**{**lkw, "loc": "upper right"})

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig_fc)
    print(f"Figure saved -> {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="Robot locomotion accuracy analysis (Motive CSV)"
    )
    parser.add_argument("--track", type=str, default="track.csv")
    parser.add_argument("--robot", type=str, default="robot.csv")
    parser.add_argument("--output", type=str, default="locomotion_analysis.png")
    parser.add_argument(
        "--track_body",
        type=str,
        default=None,
        help="Rigid body name for the track (None = single-body CSV fallback)",
    )
    parser.add_argument(
        "--robot_body",
        type=str,
        default="Robot",
        help="Rigid body name for the robot in a multi-body CSV",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Coordinate scale factor (e.g. 0.001 converts mm to m)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=60,
        help="Keep every Nth robot frame (1 = no downsampling)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_step",
        type=float,
        default=MAX_STEP_MM,
        help="Max plausible robot move per frame (mm); larger single-frame steps "
        "are tracking teleports (not drawn as bridges; used to find end-of-run junk)",
    )
    parser.add_argument(
        "--idle_radius",
        type=float,
        default=IDLE_RADIUS_MM,
        help="Settle radius (mm) for the trailing-run trim",
    )
    parser.add_argument(
        "--idle_seconds",
        type=float,
        default=IDLE_MIN_S,
        help="Min dwell (s) before a trailing parked tail is trimmed",
    )
    parser.add_argument(
        "--no_trim_tail",
        action="store_true",
        help="Keep the end-of-run tail (idle dwell / off-track lurch) instead of trimming it",
    )
    parser.add_argument(
        "--no_align",
        action="store_true",
        help="Skip coordinate-system alignment between takes",
    )
    parser.add_argument(
        "--start_anchor",
        action="store_true",
        help="Align by shifting the robot so its start sits exactly on the "
        "track's start, instead of the default whole-path best fit",
    )
    parser.add_argument(
        "--no_sync_start",
        action="store_true",
        help="Skip trimming the robot's lead-in to the track's starting position",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use light matplotlib theme instead of dark",
    )
    args = parser.parse_args()

    track_path = Path(args.track)
    robot_path = Path(args.robot)

    track_df = load_motive_csv(track_path, body_name=args.track_body, scale=args.scale)
    track_dur = track_df["time"].iloc[-1] - track_df["time"].iloc[0]
    print(f"Track loaded : {len(track_df)} frames, {track_dur:.2f} s duration")

    if not robot_path.exists():
        if robot_path != Path("robot.csv"):
            raise FileNotFoundError(
                f"{robot_path} not found. Mock data is only auto-generated for "
                "the default 'robot.csv' path to avoid overwriting real recordings."
            )
        print(f"{robot_path} not found — generating mock robot data ...")
        rng = np.random.default_rng(args.seed)
        robot_df = generate_mock_robot(track_df, robot_path, rng)
        print(f"Mock robot CSV written -> {robot_path}")
    else:
        robot_df = load_motive_csv(
            robot_path, body_name=args.robot_body, scale=args.scale
        )

    if not args.no_trim_tail:
        robot_df = _trim_trailing_run(
            robot_df,
            max_step=args.max_step,
            radius=args.idle_radius,
            min_seconds=args.idle_seconds,
        )

    # capture surviving mid-run teleport times on the full-res path, so the plot
    # can break the line there without confusing striding for a teleport.
    jump_times = robot_df["time"].to_numpy()[
        _jump_break_indices(robot_df[["x", "z"]].to_numpy(), args.max_step)
    ]

    robot_dur = robot_df["time"].iloc[-1] - robot_df["time"].iloc[0]
    print(f"Robot loaded : {len(robot_df)} frames, {robot_dur:.2f} s duration")

    if args.stride > 1:
        robot_df = robot_df.iloc[:: args.stride].reset_index(drop=True)
        print(f"Robot downsampled: {len(robot_df)} frames (stride {args.stride})")

    if args.no_align:
        print("Alignment skipped (--no_align)")
    elif args.start_anchor:
        robot_df = align_start_anchor(track_df, robot_df)
    else:
        robot_df = align_robot_to_track(track_df, robot_df)

    if not args.no_sync_start:
        robot_df = _align_robot_start(track_df, robot_df)

    nn_errors = compute_deviation(track_df, robot_df)

    print_summary(nn_errors)

    track_plot = track_df.iloc[:: args.stride].reset_index(drop=True)
    out = Path(args.output)
    n = 1
    while True:
        candidate = out.with_name(f"{out.stem}_{n}{out.suffix}")
        if not candidate.exists():
            break
        n += 1

    plot_all(
        track_plot,
        robot_df,
        nn_errors,
        candidate,
        dark=not args.light,
        jump_times=jump_times,
    )


if __name__ == "__main__":
    main()