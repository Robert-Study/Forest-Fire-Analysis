import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import curve_fit

EMPTY, TREE, FIRE = 0, 1, 2

PIXEL_SCALE = 20

TREE_RGB = (0, 180, 0)
FIRE_RGB = (255, 0, 0)
TREE_COLOR = "#00B400"
FIRE_COLOR = "#FF0000"
CRIT_LINE_COLOR = "#FFD400"

R2_MIN = 0.80

FAST_ANIM_BATCH = 5
NO_ANIM_BATCH = 50

PLOT_EVERY_ANIM = 1
PLOT_EVERY_FAST_ANIM = 5
PLOT_EVERY_NO_ANIM = 5

CMAP = np.array([[0, 0, 0], [*TREE_RGB], [*FIRE_RGB]], dtype=np.uint8)

REPORT_URL = "https://1drv.ms/w/c/4a8cd531de3d2eb8/IQBZnVfntMDyQJlJ3XZqlQ6jAfGht4bdf8ZaGZfG3lK7YNw?e=bAkuSJ"


# ------------------------- small helpers -------------------------
def fmt_val_err(val: float, err: float, sig: int = 3) -> str:
    if not np.isfinite(val) or not np.isfinite(err):
        return f"{val} ± {err}"
    v_str = f"{val:.{sig}g}"
    if "e" in v_str or "E" in v_str:
        e_str = f"{err:.{sig}g}"
        return f"{v_str} ± {e_str}"
    decimals = len(v_str.split(".")[1]) if "." in v_str else 0
    e_str = f"{err:.{decimals}f}"
    return f"{v_str} ± {e_str}"


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ncrit_from_ignition(ignition: float) -> int:
    ign = max(float(ignition), 1e-30)
    ncrit = int(24.063984 * (ign ** (-0.398039)))
    return max(1, ncrit)


def round_up(x: float, step: float) -> float:
    if x <= 0:
        return step
    return float(step * np.ceil(float(x) / float(step)))


# ------------------------- initial tree model + parsing -------------------------
def initial_tree_coverage_model(p: float, f: float) -> float:
    """
    C = 11.03 * p^(-0.01138) * f^(-0.07702)
    Interpreted here as initial tree coverage (%), clipped to [0, 100].
    """
    p = float(p)
    f = float(f)
    if p <= 0.0 or f <= 0.0:
        return 0.0
    C = 11.03 * (p ** (-0.01138)) * (f ** (-0.07702))
    return float(np.clip(C, 0.0, 100.0))


def parse_initial_trees_text(txt: str):
    """
    Returns (is_auto: bool, coverage_pct: float or None, note: str or None)
    Accepts: "Auto" or numeric percent.
    """
    s = (txt or "").strip().lower()
    if s in ("auto", "a"):
        return True, None, None
    try:
        v = float(s)
        return False, float(np.clip(v, 0.0, 100.0)), None
    except Exception:
        return False, 0.0, "Unrecognised initial tree input → using 0."


def seed_initial_trees(ss, coverage_pct: float):
    """Seed the board with an initial tree fraction (coverage_pct in [0,100])."""
    L = int(ss.locked_L)
    total = L * L
    cov = float(np.clip(coverage_pct, 0.0, 100.0))
    n_trees = int(round((cov / 100.0) * total))
    if n_trees <= 0:
        ss.tree_count = 0
        ss.fire_count = 0
        ss.tree[0] = 0
        ss.burning[0] = 0
        return

    idx = np.random.choice(total, size=n_trees, replace=False)
    ys = (idx // L).astype(int)
    xs = (idx % L).astype(int)
    ss.board[ys, xs] = TREE

    ss.tree_count = n_trees
    ss.fire_count = 0
    ss.tree[0] = n_trees
    ss.burning[0] = 0


# ------------------------- duration & seed parsing (Part 2) -------------------------
def parse_duration_text(txt: str):
    """
    Returns (is_auto: bool, frames: int or None, note: str or None)
    Accepts: "Auto" / "Autoset" (case-insensitive) or integer text.
    """
    s = (txt or "").strip().lower()
    if s in ("auto", "autoset", "a"):
        return True, None, None
    try:
        v = int(float(s))
        return False, max(1, v), None
    except Exception:
        return True, None, "Unrecognised duration input → using Auto."


def parse_seed_text(txt: str):
    """
    Returns (is_random: bool, seed_value: int or None, note: str or None)
    Accepts: "Random" or integer text.
    If random: seed_value is None and we generate a fresh seed at Run time.
    """
    s = (txt or "").strip().lower()
    if s in ("random", "rand", "r"):
        return True, None, None
    try:
        v = int(float(s))
        v = int(np.clip(v, 0, 2**31 - 1))
        return False, v, None
    except Exception:
        return True, None, "Unrecognised seed input → using Random."


def generate_fresh_seed() -> int:
    ssq = np.random.SeedSequence()
    seed_u32 = int(ssq.generate_state(1, dtype=np.uint32)[0])
    seed = int(seed_u32 % (2**31 - 1))
    return seed


# ------------------------- decay model -------------------------
def decaying_sine(t, C, A, d, w, phi):
    return C + A * np.exp(-d * t) * np.sin(w * t + phi)


def r2_score(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)


def fit_decaying_sine_to_signal(y_pct: np.ndarray, warmup_frames: int):
    N = int(len(y_pct))
    if N < 30:
        raise ValueError("Not enough data to fit.")
    crop = int(max(0, min(int(warmup_frames), N - 10)))
    t_fit = np.arange(N - crop)
    y_fit = y_pct[crop:]
    bounds = ([0, 0, 0, 0, -np.pi], [100, 100, 1, 1, np.pi])
    params, cov = curve_fit(decaying_sine, t_fit, y_fit, bounds=bounds, maxfev=20000)
    errors = np.sqrt(np.diag(cov))
    y_model = decaying_sine(t_fit, *params)
    r2 = r2_score(y_fit, y_model)
    return params, errors, crop, y_model, r2


def make_fit_plot_dark(title, y_full, crop, y_model, y_label, y_min, y_max, y_tick_step, line_color, x_max):
    t_full = np.arange(len(y_full))
    t_fit = np.arange(len(y_model))
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for s in ax.spines.values():
        s.set_color("#b3b3b3")
    ax.tick_params(colors="#d0d0d0")
    ax.grid(alpha=0.25)
    ax.set_title(title, color="#e6e6e6")
    ax.set_xlabel("Time (frames)", color="#d0d0d0")
    ax.set_ylabel(y_label, color=line_color)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1e-9, y_tick_step))
    ax.set_xlim(0, max(1, int(x_max)))
    ax.plot(t_full, y_full, color=line_color, lw=2, alpha=0.35, label="Data")
    ax.plot(t_fit + crop, y_model, color="#e6e6e6", lw=1.6, ls="--", alpha=0.90, label="Fit")
    leg = ax.legend(loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("#0e1117")
    leg.get_frame().set_edgecolor("#b3b3b3")
    for txt in leg.get_texts():
        txt.set_color("#e6e6e6")
    fig.tight_layout()
    return fig


# ------------------------- population plot (dynamic axes + aligned ticks) -------------------------
def make_population_plot(tree, burn, grid_size, frames, t, ncrit_line=None):
    total = float(grid_size * grid_size)
    x = np.arange(t + 1)
    tree_pct = 100.0 * tree[: t + 1] / total
    burn_pct = 100.0 * burn[: t + 1] / total

    # dynamic maxima
    tree_max = round_up(float(np.max(tree_pct)), 10.0)
    tree_max = float(np.clip(tree_max, 10.0, 100.0))

    burn_max = round_up(float(np.max(burn_pct)), 1.0)
    burn_max = float(max(1.0, burn_max))  # at least 1%

    tree_ticks = np.arange(0.0, tree_max + 1e-9, 10.0)
    # align burning ticks to the same gridlines (same y positions in axes coords)
    pos = tree_ticks / tree_max
    burn_ticks = np.clip(np.round(pos * burn_max), 0.0, burn_max)  # nearest 1%
    # ensure endpoints
    burn_ticks[0] = 0.0
    burn_ticks[-1] = burn_max

    fig, ax1 = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax1.set_facecolor("#0e1117")
    for s in ax1.spines.values():
        s.set_color("#b3b3b3")
    ax1.tick_params(colors="#d0d0d0")
    ax1.grid(alpha=0.25)

    # remove plot title (heading exists above in UI)
    ax1.set_title("", color="#e6e6e6")
    ax1.set_xlabel("Time (frames)", color="#d0d0d0")
    ax1.set_ylabel("Tree coverage (%)", color=TREE_COLOR)
    ax1.set_xlim(0, int(frames))
    ax1.set_ylim(0, tree_max)
    ax1.set_yticks(tree_ticks)
    ax1.plot(x, tree_pct, color=TREE_COLOR, lw=2, label="Tree coverage (%)")

    show_crit_line = (ncrit_line is not None) and (float(ncrit_line) < float(frames))
    if show_crit_line:
        ax1.axvline(
            float(ncrit_line),
            linestyle="--",
            linewidth=1.6,
            alpha=0.5,
            color=CRIT_LINE_COLOR,
            label="Critical Region",
        )

    ax2 = ax1.twinx()
    for s in ax2.spines.values():
        s.set_color("#b3b3b3")
    ax2.tick_params(colors="#d0d0d0")
    ax2.set_ylabel("Burning (%)", color=FIRE_COLOR)
    ax2.set_ylim(0, burn_max)
    ax2.set_yticks(burn_ticks)
    ax2.plot(x, burn_pct, color=FIRE_COLOR, lw=1.6, alpha=0.45, label="Burning (%)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("#0e1117")
    leg.get_frame().set_edgecolor("#b3b3b3")
    for txt in leg.get_texts():
        txt.set_color("#e6e6e6")

    fig.tight_layout()
    return fig


# ------------------------- rendering -------------------------
def board_to_rgb_upscaled(board):
    rgb = CMAP[board]
    return np.kron(rgb, np.ones((PIXEL_SCALE, PIXEL_SCALE, 1), dtype=np.uint8))


def board_to_rgb_upscaled_id(board, fire_id, fire_colors):
    frame = np.zeros((board.shape[0], board.shape[1], 3), dtype=np.uint8)
    frame[board == TREE] = np.array(TREE_RGB, dtype=np.uint8)
    ys, xs = np.where(board == FIRE)
    if ys.size:
        fids = fire_id[ys, xs].astype(np.int64)
        uniq = np.unique(fids)
        for fid in uniq.tolist():
            if fid < 0:
                continue
            if fid not in fire_colors:
                r = int(np.random.randint(50, 256))
                g = int(np.random.randint(0, 160))
                b = int(np.random.randint(50, 256))
                fire_colors[fid] = np.array([r, g, b], dtype=np.uint8)
        for y, x, fid in zip(ys, xs, fids.tolist()):
            frame[y, x] = fire_colors.get(fid, np.array(FIRE_RGB, dtype=np.uint8))
    return np.kron(frame, np.ones((PIXEL_SCALE, PIXEL_SCALE, 1), dtype=np.uint8))


# ------------------------- simulation updates -------------------------
def update_forest_counts_noid(board, tree_count, fire_count, growth, ignition):
    from forest_fire.core_simulations.fast_simulation import update_forest
    new = update_forest(board, growth, ignition)
    return new, int(np.count_nonzero(new == TREE)), int(np.count_nonzero(new == FIRE))


def update_forest_counts_id(board, fire_id, next_id, tree_count, fire_count, growth, ignition, use_uint32=False):
    from forest_fire.core_simulations.id_simulation import update_forest_ID
    bits = 32 if use_uint32 or any(0 < x < 1 / 65536 for x in (growth, ignition)) else 16
    new, ids, next_id = update_forest_ID(board, fire_id, growth, ignition, next_id, 0, probability_bits=bits)
    return new, ids, next_id, int(np.count_nonzero(new == TREE)), int(np.count_nonzero(new == FIRE))


# ------------------------- fire size statistics -------------------------
def size_counts_no_binning(sizes: np.ndarray):
    sizes = sizes.astype(int)
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return None, None
    m = int(np.max(sizes))
    counts = np.bincount(sizes, minlength=m + 1)
    x = np.arange(1, m + 1)
    y = counts[1:]
    mask = y > 0
    return x[mask].astype(float), y[mask].astype(float)


def powerlaw_model(s, A, alpha):
    return A * (s ** (-alpha))


def truncated_powerlaw_model(s, A, alpha, c):
    return A * (s ** (-alpha)) * np.exp(-s / c)


def fit_powerlaw(s_fit, c_fit):
    params, cov = curve_fit(powerlaw_model, s_fit, c_fit, p0=[1.0, 1.0], maxfev=200000)
    A, alpha = params
    yhat = powerlaw_model(s_fit, A, alpha)
    r2 = r2_score(c_fit, yhat)
    errs = np.sqrt(np.diag(cov)) if cov.size else np.array([np.nan, np.nan])
    return float(A), float(alpha), float(errs[0]), float(errs[1]), float(r2)


def fit_truncated_powerlaw(s_fit, c_fit):
    lower = [0.0, 0.0, 1.0]
    upper = [np.inf, np.inf, np.inf]
    params, cov = curve_fit(
        truncated_powerlaw_model,
        s_fit,
        c_fit,
        bounds=(lower, upper),
        maxfev=400000,
    )
    A, alpha, c = params
    yhat = truncated_powerlaw_model(s_fit, A, alpha, c)
    r2 = r2_score(c_fit, yhat)
    errs = np.sqrt(np.diag(cov)) if cov.size else np.array([np.nan, np.nan, np.nan])
    return float(A), float(alpha), float(c), float(errs[0]), float(errs[1]), float(errs[2]), float(r2)


def plot_counts_with_two_models(sizes: np.ndarray, L: int):
    x, y = size_counts_no_binning(sizes)
    if x is None or x.size < 10:
        return None, {"power": None, "trunc": None}, "Not enough completed fires after the transient."

    # Event size is a count of burned cells, not the linear grid width.

    fits = {"power": None, "trunc": None}

    fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for s in ax.spines.values():
        s.set_color("#b3b3b3")
    ax.tick_params(colors="#d0d0d0")
    ax.grid(alpha=0.25)

    # ---- gradient point size + opacity (min size: 2x + 1.0 opacity; max size: 1x + 0.5 opacity)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        norm = np.zeros_like(x, dtype=float)
    else:
        norm = (np.log(x) - np.log(xmin)) / (np.log(xmax) - np.log(xmin))
        norm = np.clip(norm, 0.0, 1.0)

    size_mult = 2.0 - norm  # 2 -> 1
    alpha_vals = 1.0 - 0.5 * norm  # 1 -> 0.5
    sizes_pts = 20.0 * size_mult

    colors = [(230 / 255, 230 / 255, 230 / 255, float(a)) for a in alpha_vals]
    ax.scatter(x, y, s=sizes_pts, c=colors, marker="o", label="Data")

    x_ref = np.logspace(np.log10(max(1.0, x.min())), np.log10(x.max()), 400)

    try:
        A1, a1, _, _, r2p = fit_powerlaw(x, y)
        fits["power"] = (A1, a1, r2p)
        if np.isfinite(r2p) and r2p >= R2_MIN:
            ax.plot(
                x_ref,
                powerlaw_model(x_ref, A1, a1),
                linestyle="--",
                linewidth=2.2,
                color="#D90429",
                label=f"Power law (A={A1:.2g}, α={a1:.2g}, R²={r2p:.3f})",
            )
    except Exception:
        fits["power"] = None

    # truncated: same long dashes as power law, dark green
    try:
        A2, a2, c, _, _, _, r2t = fit_truncated_powerlaw(x, y)
        fits["trunc"] = (A2, a2, c, r2t)
        if np.isfinite(r2t) and r2t >= R2_MIN:
            ax.plot(
                x_ref,
                truncated_powerlaw_model(x_ref, A2, a2, c),
                linestyle="--",
                linewidth=2.2,
                color="#006400",
                label=f"Truncated power law (A={A2:.2g}, α={a2:.2g}, c={c:.2g}, R²={r2t:.3f})",
            )
    except Exception:
        fits["trunc"] = None

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, max(2.0, float(x.max()) * 1.05))
    ax.set_xlabel("Fire size (s)", color="#d0d0d0")
    ax.set_ylabel("Count N(s)", color="#d0d0d0")
    ax.set_title("Counts of different fire sizes (post-transient window)", color="#e6e6e6")

    leg = ax.legend(frameon=True)
    leg.get_frame().set_facecolor("#0e1117")
    leg.get_frame().set_edgecolor("#b3b3b3")
    for txt in leg.get_texts():
        txt.set_color("#e6e6e6")

    fig.tight_layout()
    return fig, fits, None


def compute_perimeter_from_cells(cells: np.ndarray, L: int) -> int:
    if cells.size == 0:
        return 0
    ys = (cells // L).astype(int)
    xs = (cells % L).astype(int)
    mask = np.zeros((L, L), dtype=bool)
    mask[ys, xs] = True
    per = 0
    by, bx = np.where(mask)
    for y, x in zip(by.tolist(), bx.tolist()):
        per += 0 if mask[(y - 1) % L, x] else 1
        per += 0 if mask[(y + 1) % L, x] else 1
        per += 0 if mask[y, (x - 1) % L] else 1
        per += 0 if mask[y, (x + 1) % L] else 1
    return int(per)


# ------------------------- session state -------------------------
def init_defaults():
    ss = st.session_state
    ss.setdefault("sim_state", "config")
    ss.setdefault("locked_mode", "Manual")

    ss.setdefault("grid_size", 20)
    ss.setdefault("growth_prob", 0.0)
    ss.setdefault("ignition_prob", 0.0)

    ss.setdefault("enable_animation", False)
    ss.setdefault("fire_colours", "Classic")

    # Population plot ON by default (all presets too)
    ss.setdefault("plot_population", True)
    ss.setdefault("plot_decay", False)
    ss.setdefault("plot_size_counts", False)
    ss.setdefault("plot_size_perimeter", False)

    # advanced
    ss.setdefault("disable_fire_ids", False)
    ss.setdefault("use_uint32", False)

    # Part 2 inputs
    ss.setdefault("duration_text", "Auto")
    ss.setdefault("seed_text", "Random")
    ss.setdefault("seed_last_used", None)

    # initiation time (formerly warmup)
    ss.setdefault("initiation_time", 20)

    # initial trees: default OFF (0)
    ss.setdefault("initial_tree_text", "0")
    ss.setdefault("initial_tree_last_used", None)

    ss.setdefault("pending_preset", None)
    ss.setdefault("anim_cap_msg", False)

    ss.setdefault("board", None)
    ss.setdefault("t", 0)

    ss.setdefault("plot_fig", None)
    ss.setdefault("plot_t", -1)

    ss.setdefault("fit_tree_fig", None)
    ss.setdefault("fit_burning_fig", None)
    ss.setdefault("fit_tree_msg", None)
    ss.setdefault("fit_burning_msg", None)
    ss.setdefault("fit_params_df", None)
    ss.setdefault("fit_phase_heading", None)


def clear_fit_outputs(ss):
    ss.fit_tree_fig = None
    ss.fit_burning_fig = None
    ss.fit_tree_msg = None
    ss.fit_burning_msg = None
    ss.fit_params_df = None
    ss.fit_phase_heading = None


def apply_preset_config(preset: str):
    ss = st.session_state

    # All presets include population plot ON by default
    ss.plot_population = True

    if preset == "Animation":
        ss.grid_size = 50
        ss.growth_prob = 0.03
        ss.ignition_prob = 0.001
        ss.duration_text = "50"

        ss.enable_animation = True
        ss.fire_colours = "Classic"

        ss.plot_decay = False
        ss.plot_size_counts = False
        ss.plot_size_perimeter = False

        ss.disable_fire_ids = False
        ss.use_uint32 = False
        ss.initiation_time = 20

        ss.initial_tree_text = "0"

    elif preset == "Decay model":
        ss.grid_size = 2000
        ss.growth_prob = 0.02
        ss.ignition_prob = 0.001
        ss.duration_text = "Auto"

        ss.enable_animation = False
        ss.fire_colours = "Classic"

        ss.plot_decay = True
        ss.plot_size_counts = False
        ss.plot_size_perimeter = False

        ss.disable_fire_ids = True
        ss.use_uint32 = False
        ss.initiation_time = 20

        ss.initial_tree_text = "0"

    elif preset == "Fire size vs counts":
        ss.grid_size = 300
        ss.growth_prob = 0.003
        ss.ignition_prob = 0.0001
        ss.duration_text = "Auto"

        ss.enable_animation = False
        ss.fire_colours = "Unique (ID)"

        ss.plot_decay = False
        ss.plot_size_counts = True
        ss.plot_size_perimeter = False

        ss.disable_fire_ids = False
        ss.use_uint32 = False
        ss.initiation_time = 20

        # Auto initial trees ON for counts
        ss.initial_tree_text = "Auto"

    elif preset == "Fire size vs perimeter":
        # requested parameters
        ss.grid_size = 200
        ss.growth_prob = 1e-5
        ss.ignition_prob = 5e-8
        ss.duration_text = "Auto"

        ss.enable_animation = False
        ss.fire_colours = "Unique (ID)"
        ss.plot_decay = False
        ss.plot_size_counts = False
        ss.plot_size_perimeter = True

        ss.disable_fire_ids = False
        ss.use_uint32 = True
        ss.initiation_time = 20

        # Auto initial trees ON for perimeter
        ss.initial_tree_text = "Auto"
   	 # --- FIXED SEED FOR THIS PRESET ---
        ss.seed_text = "1518689727"
        ss.use_seed = True
        ss.seed = 1518689727
        ss.resolved_seed = 1518689727


def enforce_ui_constraints(ss):
    ss.anim_cap_msg = False
    if bool(ss.enable_animation) and int(ss.grid_size) > 250:
        ss.grid_size = 250
        ss.anim_cap_msg = True

    if bool(ss.plot_size_perimeter):
        ss.use_uint32 = True

    require_ids = bool(ss.enable_animation) or bool(ss.plot_size_counts) or bool(ss.plot_size_perimeter)
    if require_ids:
        ss.disable_fire_ids = False

    ss.initiation_time = int(max(0, min(int(ss.initiation_time), 200)))


# ------------------------- run setup + stepping -------------------------
def start_run_from_current_settings(ss, mode_name: str, fast_animation: bool):
    L = int(ss.grid_size)
    p = float(ss.growth_prob)
    f = float(ss.ignition_prob)

    # Duration
    is_auto, frames_val, _ = parse_duration_text(ss.duration_text)
    ncrit = ncrit_from_ignition(f) if f > 0 else 1

    if is_auto:
        need_critical = bool(ss.plot_size_counts or ss.plot_size_perimeter)
        # requested: 1.2N for normal, 2N for critical
        frames = int((2.0 if need_critical else 1.2) * ncrit)
    else:
        frames = int(frames_val if frames_val is not None else int(1.2 * ncrit))

    # Seed (fresh every Run if Random)
    is_rand, seed_val, _ = parse_seed_text(ss.seed_text)
    seed_used = int(generate_fresh_seed() if is_rand else (seed_val if seed_val is not None else generate_fresh_seed()))
    ss.seed_last_used = seed_used
    np.random.seed(seed_used)

    enable_anim = bool(ss.enable_animation)
    plot_pop = bool(ss.plot_population)
    plot_decay = bool(ss.plot_decay)
    plot_counts = bool(ss.plot_size_counts)
    plot_perim = bool(ss.plot_size_perimeter)

    track_ids = not bool(ss.disable_fire_ids)
    if mode_name == "Decay model":
        track_ids = False
    if enable_anim and (ss.fire_colours == "Unique (ID)"):
        track_ids = True
    if plot_counts or plot_perim:
        track_ids = True

    use_uint32 = bool(ss.use_uint32) or bool(plot_perim)

    ss.locked_mode = mode_name
    ss.sim_state = "running"

    ss.locked_L = L
    ss.locked_p = p
    ss.locked_f = f
    ss.locked_ncrit = int(ncrit)
    ss.locked_frames = int(frames)

    ss.locked_enable_anim = enable_anim
    ss.locked_fast_anim = bool(fast_animation)
    ss.locked_use_uint32 = bool(use_uint32)

    ss.locked_plot_pop = plot_pop
    ss.locked_plot_decay = plot_decay
    ss.locked_plot_counts = plot_counts
    ss.locked_plot_perim = plot_perim

    ss.locked_track_ids = bool(track_ids)

    ss.t = 0
    ss.board = np.zeros((L, L), dtype=np.uint8)
    ss.tree = np.zeros(frames + 1, dtype=np.int32)
    ss.burning = np.zeros(frames + 1, dtype=np.int32)
    ss.tree_count = 0
    ss.fire_count = 0

    ss.plot_fig = None
    ss.plot_t = -1
    clear_fit_outputs(ss)

    ss.fire_id = None
    ss.next_fire_id = 0
    ss.fire_colors = None
    ss.fire_ignition = None
    ss.fire_total_burn = None
    ss.fire_burned_cells = None
    ss.fire_perimeter = None

    if ss.locked_track_ids:
        ss.fire_id = np.full((L, L), -1, dtype=np.int32)
        ss.next_fire_id = 0
        ss.fire_colors = {}
        ss.fire_ignition = {}
        ss.fire_total_burn = {}
        if ss.locked_plot_perim:
            ss.fire_burned_cells = {}

    # Initial trees (Auto or custom, independent of duration)
    auto_init, cov_val, _ = parse_initial_trees_text(ss.initial_tree_text)
    if auto_init:
        cov = float(initial_tree_coverage_model(p, f))
        ss.initial_tree_last_used = cov
    else:
        cov = float(cov_val if cov_val is not None else 0.0)
        ss.initial_tree_last_used = None

    seed_initial_trees(ss, cov)


def maybe_update_population_plot(ss):
    if not ss.locked_plot_pop:
        return
    t = int(ss.t)
    if ss.locked_enable_anim:
        plot_every = PLOT_EVERY_FAST_ANIM if ss.locked_fast_anim else PLOT_EVERY_ANIM
    else:
        plot_every = PLOT_EVERY_NO_ANIM

    due = (ss.plot_fig is None) or (t == 0) or (t >= ss.locked_frames) or ((t - ss.plot_t) >= plot_every)
    if not due:
        return

    if ss.plot_fig is not None:
        try:
            plt.close(ss.plot_fig)
        except Exception:
            pass

    ss.plot_fig = make_population_plot(
        ss.tree,
        ss.burning,
        ss.locked_L,
        ss.locked_frames,
        t,
        ncrit_line=int(ss.locked_ncrit),
    )
    ss.plot_t = t


def step_once(ss):
    if ss.locked_track_ids:
        board, fid, next_id, tree_count, fire_count = update_forest_counts_id(
            ss.board,
            ss.fire_id,
            ss.next_fire_id,
            ss.tree_count,
            ss.fire_count,
            ss.locked_p,
            ss.locked_f,
            use_uint32=ss.locked_use_uint32,
        )
        ss.board = board
        ss.fire_id = fid
        ss.next_fire_id = next_id
        ss.tree_count = tree_count
        ss.fire_count = fire_count

        ss.t += 1
        ss.tree[ss.t] = tree_count
        ss.burning[ss.t] = fire_count

        active = fid[fid >= 0]
        if active.size:
            ids, counts = np.unique(active, return_counts=True)
            for i, c in zip(ids.tolist(), counts.tolist()):
                if i not in ss.fire_ignition:
                    ss.fire_ignition[i] = ss.t
                    ss.fire_total_burn[i] = 0
                    if ss.locked_plot_perim:
                        ss.fire_burned_cells[i] = set()
                ss.fire_total_burn[i] += int(c)

        if ss.locked_plot_perim:
            ys, xs = np.where(board == FIRE)
            if ys.size:
                fids = fid[ys, xs].astype(np.int64)
                lin = (ys.astype(np.int64) * ss.locked_L + xs.astype(np.int64)).tolist()
                for cell, fidv in zip(lin, fids.tolist()):
                    if fidv >= 0 and fidv in ss.fire_burned_cells:
                        ss.fire_burned_cells[fidv].add(int(cell))
    else:
        board, tree_count, fire_count = update_forest_counts_noid(
            ss.board,
            ss.tree_count,
            ss.fire_count,
            ss.locked_p,
            ss.locked_f,
        )
        ss.board = board
        ss.tree_count = tree_count
        ss.fire_count = fire_count

        ss.t += 1
        ss.tree[ss.t] = tree_count
        ss.burning[ss.t] = fire_count


def finalize_perimeters(ss):
    if not ss.locked_plot_perim or ss.fire_burned_cells is None:
        return
    L = int(ss.locked_L)
    ss.fire_perimeter = {}
    for fid, s in ss.fire_burned_cells.items():
        arr = np.fromiter(s, dtype=np.int64)
        ss.fire_perimeter[fid] = compute_perimeter_from_cells(arr, L)


def do_decay_fit_at_end(ss):
    if not ss.locked_plot_decay:
        return
    t_now = int(ss.t)
    if t_now < 30:
        return

    grid = int(ss.locked_L)
    total = float(grid * grid)

    tree_pct_full = 100.0 * ss.tree[: t_now + 1].astype(float) / total
    burning_pct_full = 100.0 * ss.burning[: t_now + 1].astype(float) / total

    # Fit only up to Ncrit
    ncrit = int(ss.locked_ncrit)
    fit_len = int(min(len(tree_pct_full), ncrit + 1))
    x_max = fit_len - 1

    tree_pct = tree_pct_full[:fit_len]
    burning_pct = burning_pct_full[:fit_len]

    warmup = int(max(0, min(int(ss.initiation_time), 200)))

    tree_out = None
    burning_out = None

    try:
        params, errs, crop, y_model, r2 = fit_decaying_sine_to_signal(tree_pct, warmup_frames=warmup)
        if not np.isfinite(r2) or r2 < R2_MIN:
            msg = f"Tree fit unavailable for these parameters (R²={r2:.3f} < {R2_MIN:.2f})."
            if grid <= 500:
                msg += " Recommendation: try size > 500."
            ss.fit_tree_msg = msg
        else:
            ss.fit_tree_fig = make_fit_plot_dark(
                "Tree (%) — Decaying Sine Fit",
                tree_pct,
                crop,
                y_model,
                "Tree coverage (%)",
                0,
                100,
                10,
                TREE_COLOR,
                x_max=x_max,
            )
            tree_out = (params, errs, r2)
    except Exception:
        ss.fit_tree_msg = "Tree fit unavailable for these parameters."

    try:
        params, errs, crop, y_model, r2 = fit_decaying_sine_to_signal(burning_pct, warmup_frames=warmup)
        if not np.isfinite(r2) or r2 < R2_MIN:
            msg = f"Burning fit unavailable for these parameters (R²={r2:.3f} < {R2_MIN:.2f})."
            if grid <= 500:
                msg += " Recommendation: try size > 500."
            ss.fit_burning_msg = msg
        else:
            ss.fit_burning_fig = make_fit_plot_dark(
                "Burning (%) — Decaying Sine Fit",
                burning_pct,
                crop,
                y_model,
                "Burning (%)",
                0,
                10,
                1,
                FIRE_COLOR,
                x_max=x_max,
            )
            burning_out = (params, errs, r2)
    except Exception:
        ss.fit_burning_msg = "Burning fit unavailable for these parameters."

    if tree_out is not None and burning_out is not None:
        (pT, eT, rT) = tree_out
        (pB, eB, rB) = burning_out

        C_T, A_T, d_T, w_T, phi_T = [float(x) for x in pT]
        C_B, A_B, d_B, w_B, phi_B = [float(x) for x in pB]
        eC_T, eA_T, ed_T, ew_T, ephi_T = [float(x) for x in eT]
        eC_B, eA_B, ed_B, ew_B, ephi_B = [float(x) for x in eB]

        delta_phi = wrap_to_pi(phi_T - phi_B)
        sigma_delta_phi = float(np.sqrt((ephi_T ** 2) + (ephi_B ** 2)))

        ss.fit_phase_heading = f"Phase difference: Δφ = {delta_phi:.3g} ± {sigma_delta_phi:.2g} rad"

        df = pd.DataFrame(
            [
                {
                    "Series": "Tree Population",
                    "C": fmt_val_err(C_T, eC_T),
                    "A": fmt_val_err(A_T, eA_T),
                    "d": fmt_val_err(d_T, ed_T),
                    "ω": fmt_val_err(w_T, ew_T),
                    "φ": fmt_val_err(phi_T, ephi_T),
                    "R²": f"{float(rT):.3f}",
                },
                {
                    "Series": "Burning Population",
                    "C": fmt_val_err(C_B, eC_B),
                    "A": fmt_val_err(A_B, eA_B),
                    "d": fmt_val_err(d_B, ed_B),
                    "ω": fmt_val_err(w_B, ew_B),
                    "φ": fmt_val_err(phi_B, ephi_B),
                    "R²": f"{float(rB):.3f}",
                },
            ]
        )
        ss.fit_params_df = df


def build_fire_size_arrays_critical_only(ss):
    if (not ss.locked_track_ids) or (ss.fire_total_burn is None) or (ss.fire_ignition is None):
        return np.array([], dtype=int), np.array([], dtype=int)

    from forest_fire.analysis.event_statistics import completed_fire_sizes
    active_ids = np.unique(ss.fire_id[ss.fire_id >= 0])
    return completed_fire_sizes(ss.fire_total_burn, ss.fire_ignition, active_ids,
                                int(ss.locked_ncrit), ss.fire_perimeter if ss.locked_plot_perim else None)


# ------------------------- UI callbacks -------------------------
def on_request_preset(name: str):
    st.session_state.pending_preset = name


def on_run_pause_resume():
    ss = st.session_state
    if ss.sim_state == "running":
        ss.sim_state = "paused"
        return
    if ss.sim_state == "paused":
        ss.sim_state = "running"
        return
    if ss.sim_state in ("config", "finished"):
        ss.locked_mode = "Manual"
        start_run_from_current_settings(ss, "Manual", fast_animation=False)


def on_reset():
    ss = st.session_state
    ss.sim_state = "config"
    ss.board = None
    ss.plot_fig = None
    ss.plot_t = -1
    clear_fit_outputs(ss)
    ss.fire_id = None
    ss.fire_total_burn = None
    ss.fire_ignition = None
    ss.fire_burned_cells = None
    ss.fire_perimeter = None
    ss.seed_last_used = None
    ss.initial_tree_last_used = None
    for k in list(ss.keys()):
        if k.startswith("locked_"):
            del ss[k]


# ------------------------- STREAMLIT APP -------------------------
st.set_page_config(page_title="Forest Fire Simulation", layout="wide")
st.markdown(
    """
<style>
section[data-testid="stSidebar"] * { font-size: 0.95rem; }
label[data-testid="stWidgetLabel"] p { white-space: pre-line; }
</style>
""",
    unsafe_allow_html=True,
)

# Project title
st.markdown("<h1 style='margin-bottom: 0.2rem;'>Forest Fire Simulation</h1>", unsafe_allow_html=True)

st.caption("Explore a stochastic lattice model. Post-transient sampling uses an empirical timing rule; fitted power laws are exploratory.")

init_defaults()
ss = st.session_state

if ss.pending_preset is not None:
    preset = ss.pending_preset
    ss.pending_preset = None
    apply_preset_config(preset)

enforce_ui_constraints(ss)

locked_controls = ss.sim_state in ("running", "paused", "finished")

# Progress area ONLY after run is pressed (running/paused/finished), hidden in config (and after reset)
if ss.sim_state in ("running", "paused", "finished") and ("locked_frames" in ss) and ("t" in ss):
    pcol, tcol = st.columns([1, 3])
    with pcol:
        denom = max(1, int(ss.locked_frames))
        val = min(1.0, float(ss.t) / float(denom))
        st.progress(val)
    with tcol:
        st.write(f"Progress: {int(ss.t)}/{int(ss.locked_frames)} frames")

with st.sidebar:
    top1, top2 = st.columns(2)
    run_label = "Run"
    if ss.sim_state == "running":
        run_label = "Pause"
    elif ss.sim_state == "paused":
        run_label = "Resume"
    top1.button(run_label, use_container_width=True, key="btn_run_pause_resume", on_click=on_run_pause_resume)
    top2.button("Reset", use_container_width=True, key="btn_reset", on_click=on_reset)

    st.divider()
    st.markdown("**Presets**")
    c1, c2 = st.columns(2)
    c1.button("Animation", use_container_width=True, key="preset_anim", disabled=locked_controls, on_click=on_request_preset, args=("Animation",))
    c2.button("Decay model", use_container_width=True, key="preset_decay", disabled=locked_controls, on_click=on_request_preset, args=("Decay model",))
    c3, c4 = st.columns(2)
    c3.button("Fire size vs counts", use_container_width=True, key="preset_counts", disabled=locked_controls, on_click=on_request_preset, args=("Fire size vs counts",))
    c4.button("Fire size vs perimeter", use_container_width=True, key="preset_perim", disabled=locked_controls, on_click=on_request_preset, args=("Fire size vs perimeter",))

    st.divider()
    st.markdown("**Parameters**")
    st.number_input("Size", min_value=10, max_value=2000, step=10, key="grid_size", disabled=locked_controls)
    st.number_input("Growth probability", min_value=0.0, max_value=0.05, step=1e-6, format="%.10g", key="growth_prob", disabled=locked_controls)
    st.number_input("Ignition probability", min_value=0.0, max_value=0.02, step=1e-12, format="%.12g", key="ignition_prob", disabled=locked_controls)

    st.divider()
    st.markdown("**Outputs**")
    if ss.anim_cap_msg and (not locked_controls):
        st.info("Animation is enabled: size has been capped to 250 to avoid excessive load times.")

    st.checkbox(
        "Enable animation",
        key="enable_animation",
        disabled=locked_controls,
        help="Show an animation of the lattice evolving through time, streamlit limiting effective FPS (~3fps).",
    )
    st.checkbox(
        "Population plot",
        key="plot_population",
        disabled=locked_controls,
        help="Tree and burning fractions vs time.",
    )
    st.checkbox(
        "Decaying sine fit plots",
        key="plot_decay",
        disabled=locked_controls,
        help="Fit a decaying sine model onto the population model from the initiation time to the end of oscillatory regime.",
    )
    st.checkbox(
        "Fire size\nvs Counts",
        key="plot_size_counts",
        disabled=locked_controls,
        help="Analysis of the frequency of different sizes of fires in the post-transient window.",
    )
    st.checkbox(
        "Fire size vs perimeter",
        key="plot_size_perimeter",
        disabled=locked_controls,
        help="Analysis of the size, perimeter relation of fires in the post-transient window. Typically found at ignition << growth << 1 (may require uint32 for small probabilities).",
    )

    st.divider()
    with st.expander("Advanced", expanded=False):
        require_ids = bool(ss.enable_animation) or bool(ss.plot_size_counts) or bool(ss.plot_size_perimeter)

        st.checkbox(
            "Disable fire IDs",
            key="disable_fire_ids",
            disabled=locked_controls or require_ids,
            help="Resource-intensive but required for tracking fire sizes.",
        )
        st.checkbox(
            "Use uint32 random",
            key="use_uint32",
            disabled=locked_controls or bool(ss.plot_size_perimeter),
            help="Higher resolution of probabilities than uint16, especially required for size vs perimeter (increases load times).",
        )

        st.text_input(
            "Duration (frames)",
            key="duration_text",
            disabled=locked_controls,
            help='Type "Auto" for autoset (runs simulation based on expected decay constant and simulation application), alternatively enter an integer number of frames.',
        )

        st.number_input(
            "Initiation time",
            min_value=0,
            max_value=200,
            step=1,
            key="initiation_time",
            disabled=locked_controls,
            help="Initiation time",
        )

        st.text_input(
            "Seed",
            key="seed_text",
            disabled=locked_controls,
            help='Type "Random" for a fresh seed each time you press Run, or enter an integer seed.',
        )
        if ss.seed_last_used is not None:
            st.caption(f"Last run seed: {int(ss.seed_last_used)}")

        # No subheading / no divider: initial trees directly under seed area
        st.text_input(
            "Initial tree coverage (%)",
            key="initial_tree_text",
            disabled=locked_controls,
            help="Set a custom initial tree coverage (0is default), Type Auto to start with an estimate of the equilibrium position",
        )
        if ss.initial_tree_last_used is not None:
            st.caption(f"Equilibrium population selected: {float(ss.initial_tree_last_used):.2f}%")

    st.caption(
        "Notes: Fires can wrap around edges helping to replicate an infinitely large forest, without increasing computation. "
        "On ignition each fire is assigned an ID, this is used to track the size of each fire individually, "
        "the lowest ID takes priority if multiple fires ignite the same cell."
    )

# Pre-run info (requested text)
if ss.sim_state == "config" and ss.get("board") is None:
    st.info(
        "Select a preset (or set parameters and outputs manually), then press Run."
    )

just_finished = False

if ss.sim_state == "running":
    frames = int(ss.locked_frames)

    if ss.locked_enable_anim:
        remaining = frames - ss.t
        batch = FAST_ANIM_BATCH if ss.locked_fast_anim else 1
        for _ in range(min(batch, remaining)):
            if ss.t < frames:
                step_once(ss)
    else:
        remaining = frames - ss.t
        for _ in range(min(NO_ANIM_BATCH, remaining)):
            if ss.t < frames:
                step_once(ss)

    maybe_update_population_plot(ss)

    if ss.t >= frames:
        ss.sim_state = "finished"
        if ss.locked_plot_perim:
            finalize_perimeters(ss)
        if ss.locked_plot_decay:
            do_decay_fit_at_end(ss)
        just_finished = True

if just_finished:
    st.rerun()

t = int(ss.get("t", 0))

# Layout
if hasattr(ss, "locked_enable_anim") and bool(ss.locked_enable_anim):
    left, right = st.columns([1, 1], gap="large")
    with left:
        h1, h2 = st.columns([1.0, 1.2])
        with h1:
            st.markdown("<h3 style='margin-bottom: 0.2rem;'>Animation</h3>", unsafe_allow_html=True)
        with h2:
            st.radio(
                "",
                ["Classic", "Unique (ID)"],
                key="fire_colours",
                horizontal=True,
                label_visibility="collapsed",
                disabled=(ss.sim_state == "finished"),
            )

        # removed "Frame t/frames" caption (progress already shows)
        if ss.fire_colours == "Unique (ID)" and bool(getattr(ss, "locked_track_ids", False)):
            st.image(board_to_rgb_upscaled_id(ss.board, ss.fire_id, ss.fire_colors), use_container_width=True)
        else:
            st.image(board_to_rgb_upscaled(ss.board), use_container_width=True)

    with right:
        if ss.plot_fig is not None:
            st.markdown("<h3 style='margin-bottom: 0.2rem;'>Population series</h3>", unsafe_allow_html=True)
            st.pyplot(ss.plot_fig, use_container_width=True)

        if ss.sim_state == "finished" and bool(getattr(ss, "locked_plot_decay", False)):
            st.markdown("<h3 style='margin-top: 1rem; margin-bottom: 0.2rem;'>Decaying sine fits</h3>", unsafe_allow_html=True)

            if ss.fit_tree_fig is not None:
                st.pyplot(ss.fit_tree_fig, use_container_width=True)
            elif ss.fit_tree_msg:
                st.info(ss.fit_tree_msg)

            if ss.fit_burning_fig is not None:
                st.pyplot(ss.fit_burning_fig, use_container_width=True)
            elif ss.fit_burning_msg:
                st.info(ss.fit_burning_msg)

            st.markdown(r"$y(t)=C + A\,e^{-dt}\,\sin(\omega t + \phi)$")
            st.caption("C: baseline, A: amplitude, d: decay coefficient, ω: angular frequency, φ: phase, t: time (frames).")

            if ss.fit_phase_heading:
                st.markdown(f"**{ss.fit_phase_heading}**")

            if ss.fit_params_df is not None:
                st.dataframe(ss.fit_params_df, hide_index=True, use_container_width=True)
                st.caption(f"[Computing Report (Online)]({REPORT_URL})")

        if ss.sim_state == "finished":
            if bool(getattr(ss, "locked_plot_counts", False)) or bool(getattr(ss, "locked_plot_perim", False)):
                sizes, perims = build_fire_size_arrays_critical_only(ss)
                L = int(getattr(ss, "locked_L", 0))

                if bool(getattr(ss, "locked_plot_counts", False)):
                    st.markdown("<h3 style='margin-top: 1rem; margin-bottom: 0.2rem;'>Fire size vs counts</h3>", unsafe_allow_html=True)
                    st.markdown(r"$N(s)=A\,s^{-\alpha}$  (truncation term: $e^{-s/c}$)")
                    fig, fits, msg = plot_counts_with_two_models(sizes, L=L)
                    if msg:
                        st.info(msg)
                    if fig is not None:
                        if fits.get("power") is None or (fits.get("power")[2] < R2_MIN if fits.get("power") else True):
                            st.info("Power law fit unavailable for these parameters (R² < 0.80).")
                        if fits.get("trunc") is None or (fits.get("trunc")[3] < R2_MIN if fits.get("trunc") else True):
                            st.info("Truncated power law fit unavailable for these parameters (R² < 0.80).")
                        st.pyplot(fig, use_container_width=True)

                if bool(getattr(ss, "locked_plot_perim", False)):
                    st.markdown("<h3 style='margin-top: 1rem; margin-bottom: 0.2rem;'>Fire size vs perimeter</h3>", unsafe_allow_html=True)
                    mask = (sizes > 0) & (perims > 0)
                    if np.count_nonzero(mask) < 20:
                        st.info("Not enough completed fires after the transient with perimeter data.")
                    else:
                        fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=120)
                        fig.patch.set_facecolor("#0e1117")
                        ax.set_facecolor("#0e1117")
                        for s in ax.spines.values():
                            s.set_color("#b3b3b3")
                        ax.tick_params(colors="#d0d0d0")
                        ax.grid(alpha=0.25)
                        ax.scatter(sizes[mask].astype(float), perims[mask].astype(float), s=14, alpha=0.65, marker="x")
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        # no cap at L for perimeter: set based on data
                        xmax = float(np.max(sizes[mask])) if np.count_nonzero(mask) else float(L)
                        ax.set_xlim(1, max(2.0, xmax))
                        ax.set_xlabel("Fire size (s)", color="#d0d0d0")
                        ax.set_ylabel("Perimeter", color="#d0d0d0")
                        ax.set_title("Fire size vs perimeter (post-transient window)", color="#e6e6e6")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)

else:
    # no animation view
    if ss.plot_fig is not None:
        st.markdown("<h3 style='margin-bottom: 0.2rem;'>Population series</h3>", unsafe_allow_html=True)
        st.pyplot(ss.plot_fig, use_container_width=True)

    if ss.sim_state == "finished":
        if bool(getattr(ss, "locked_plot_decay", False)):
            st.markdown("<h3 style='margin-top: 1rem; margin-bottom: 0.2rem;'>Decaying sine fits</h3>", unsafe_allow_html=True)

            if ss.fit_tree_fig is not None:
                st.pyplot(ss.fit_tree_fig, use_container_width=True)
            elif ss.fit_tree_msg:
                st.info(ss.fit_tree_msg)

            if ss.fit_burning_fig is not None:
                st.pyplot(ss.fit_burning_fig, use_container_width=True)
            elif ss.fit_burning_msg:
                st.info(ss.fit_burning_msg)

            st.markdown(r"$y(t)=C + A\,e^{-dt}\,\sin(\omega t + \phi)$")
            st.caption("C: baseline, A: amplitude, d: decay coefficient, ω: angular frequency, φ: phase, t: time (frames).")

            if ss.fit_phase_heading:
                st.markdown(f"**{ss.fit_phase_heading}**")

            if ss.fit_params_df is not None:
                st.dataframe(ss.fit_params_df, hide_index=True, use_container_width=True)

        if bool(getattr(ss, "locked_plot_counts", False)) or bool(getattr(ss, "locked_plot_perim", False)):
            sizes, perims = build_fire_size_arrays_critical_only(ss)
            L = int(getattr(ss, "locked_L", 0))

            if bool(getattr(ss, "locked_plot_counts", False)):
                st.markdown("<h3 style='margin-top: 1rem; margin-bottom: 0.2rem;'>Fire size vs counts</h3>", unsafe_allow_html=True)
                st.markdown(r"$N(s)=A\,s^{-\alpha}$  (truncation term: $e^{-s/c}$)")
                fig, fits, msg = plot_counts_with_two_models(sizes, L=L)

                if msg:
                    st.info(msg)

                if fig is not None:
                    if fits.get("power") is None or (fits.get("power")[2] < R2_MIN if fits.get("power") else True):
                        st.info("Power law fit unavailable for these parameters (R² < 0.80).")

                    if fits.get("trunc") is None or (fits.get("trunc")[3] < R2_MIN if fits.get("trunc") else True):
                        st.info("Truncated power law fit unavailable for these parameters (R² < 0.80).")

                    st.pyplot(fig, use_container_width=True)

            if bool(getattr(ss, "locked_plot_perim", False)):
                st.markdown(
                    "<h3 style='margin-top: 1rem; margin-bottom: 0.2rem;'>Fire size vs perimeter</h3>",
                    unsafe_allow_html=True,
                )

                mask = (sizes > 0) & (perims > 0)

                # ---- DEBUG BLOCK (temporary) ----
                st.write("Fires tracked (total_burn):", len(ss.fire_total_burn or {}))
                st.write("Fires with burned-cell sets:", len(ss.fire_burned_cells or {}))
                st.write("Fires with computed perimeters:", len(ss.fire_perimeter or {}))

                ncrit = int(ss.locked_ncrit)
                crit_ids = [fid for fid, t0 in (ss.fire_ignition or {}).items() if t0 > ncrit]
                st.write("Critical fires (ignition > Ncrit):", len(crit_ids))

                crit_with_per = [
                    fid for fid in crit_ids if (ss.fire_perimeter or {}).get(fid, 0) > 0
                ]
                st.write("Critical fires with perimeter > 0:", len(crit_with_per))
                # --------------------------------

                if np.count_nonzero(mask) < 10:
                    st.info("Not enough completed fires after the transient with perimeter data.")
                else:
                    fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=120)
                    fig.patch.set_facecolor("#0e1117")
                    ax.set_facecolor("#0e1117")

                    for s in ax.spines.values():
                        s.set_color("#b3b3b3")

                    ax.tick_params(colors="#d0d0d0")
                    ax.grid(alpha=0.25)

                    ax.scatter(
                        sizes[mask].astype(float),
                        perims[mask].astype(float),
                        s=14,
                        alpha=0.65,
                        marker="x",
                    )

                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    xmax = float(np.max(sizes[mask])) if np.any(mask) else float(L)
                    ax.set_xlim(1, max(2.0, xmax))

                    ax.set_xlabel("Fire size (s)", color="#d0d0d0")
                    ax.set_ylabel("Perimeter", color="#d0d0d0")
                    ax.set_title("Fire size vs perimeter (post-transient window)", color="#e6e6e6")

                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)

if ss.sim_state == "running":
    st.rerun()
