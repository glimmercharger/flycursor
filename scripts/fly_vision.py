"""fly_vision.py - what the fly sees on the screen, and a picture of its brain at work.

Vision: a screenshot is the fly's world. From the cursor's position and heading
a 1-D panorama is read off the screen (luminance along rays fanning out over the
fly's field of view), then projected onto the 6,006 photoreceptors by their
azimuth, the same recipe as flybrain.eyes.Eyes uses for its synthetic blobs.
Brightness plus brightness *change* (motion) is what the photoreceptors get.

In this spiking model the photoreceptor signal fades in the lamina (the package
author says so too), so the panorama is ALSO run through a looming detector: a
dark patch that grows wider on one side drives that side's LPLC2 looming
neurons, the route that does reach the descending neurons.

Snapshots: every N seconds a PNG is written with the screenshot (fly, heading,
field of view, food), the panorama strip it saw, and a map of all 166,700
neurons with the ones firing above the noise floor lit up by region. Rendering
takes a few seconds, so it runs in a background thread.
"""
from __future__ import annotations

import math
import os
import threading
import time
from datetime import datetime

import numpy as np
import pyautogui

# Regions of the nervous system, in fixed order, with one hue each (dark surface).
REGIONS = [
    ("optic lobes", ["ol_intrinsic", "ol_sensory", "visual_centrifugal"], "#3987e5"),
    ("visual projection", ["visual_projection"], "#d95926"),
    ("central brain", ["cb_intrinsic", "cb_endocrine", "cb_sensory_tbc", "ENS"], "#199e70"),
    ("sensory", ["cb_sensory", "vnc_sensory", "sensory_ascending", "vnc_sensory_tbc"], "#c98500"),
    ("descending (brain->body)", ["descending_neuron"], "#d55181"),
    ("ascending (body->brain)", ["ascending_neuron"], "#008300"),
    ("nerve cord", ["vnc_intrinsic", "vnc_tbc", "vnc_endocrine", "vnc_efferent"], "#9085e9"),
    ("motor", ["vnc_motor", "cb_motor"], "#e66767"),
]
OTHER = "#9a9a98"
SURFACE, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a38"
IN_USE_HZ = 3.0     # noise alone makes a neuron tick at ~1 Hz; above this it is "in use"


class Vision:
    """Reads the screen as a panorama around the fly and drives the photoreceptors."""

    def __init__(self, brain, fov_deg: float = 240.0, reach_px: float = 350.0, bins: int = 64, scale: int = 4):
        self.azimuth = brain.azimuth                        # -1 far left .. +1 far right, per photoreceptor
        self.bin_of = np.clip(np.round((self.azimuth + 1) / 2 * (bins - 1)).astype(int), 0, bins - 1)
        self.fov, self.reach, self.bins, self.scale = math.radians(fov_deg), reach_px, bins, scale
        self.rel_angles = np.linspace(-self.fov / 2, self.fov / 2, bins)
        self.left = self.rel_angles < 0
        self.radii = np.linspace(12.0, reach_px, 24)
        self.shot = None                                   # last PIL screenshot (RGB)
        self.gray = None                                   # small grayscale copy, 0..1
        self.prev = None
        self.prev_width = {"L": 0, "R": 0}
        self.streak = {"L": 0, "R": 0}                    # consecutive steps a dark patch has grown
        self.suppress = 0                                  # steps left to ignore looming after a self-move
        self.loom = {"L": 0.0, "R": 0.0}                   # LPLC2 drive per side, from the panorama
        self.lum = np.zeros(bins, np.float32)
        self.drive_vec = np.zeros(len(self.azimuth), np.float32)
        self.grabbed_at = 0.0

    def grab(self):
        self.shot = pyautogui.screenshot()
        small = self.shot.convert("L").resize((self.shot.width // self.scale, self.shot.height // self.scale))
        self.gray = np.asarray(small, np.float32) / 255.0
        self.grabbed_at = time.time()

    def panorama(self, x: float, y: float, heading: float) -> np.ndarray:
        """Mean luminance along each ray of the field of view, 0..1 per bin."""
        if self.gray is None:
            return np.full(self.bins, 0.5, np.float32)
        ang = heading + self.rel_angles                    # (bins,)
        px = (x + np.outer(np.cos(ang), self.radii)) / self.scale
        py = (y + np.outer(np.sin(ang), self.radii)) / self.scale
        h, w = self.gray.shape
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        xi = np.clip(px.astype(int), 0, w - 1)
        yi = np.clip(py.astype(int), 0, h - 1)
        lum = np.where(inside, self.gray[yi, xi], 0.0)     # off-screen is dark, like the edge of the dish
        return lum.mean(axis=1).astype(np.float32)

    def self_moved(self, steps: int = 10):
        """Tell the eyes the fly just moved itself, so the view change is not a loom."""
        self.suppress = max(self.suppress, steps)
        self.streak = {"L": 0, "R": 0}

    def drive(self, x: float, y: float, heading: float) -> np.ndarray:
        """Photoreceptor drive, 0..1, same recipe as flybrain.eyes.Eyes.drive. Also updates self.loom."""
        self.lum = self.panorama(x, y, heading)
        first = self.prev is None
        change = np.zeros_like(self.lum) if first else np.abs(self.lum - self.prev)
        self.prev = self.lum
        per_bin = np.clip(0.45 * self.lum + 1.6 * change, 0, 1)
        self.drive_vec = per_bin[self.bin_of]

        # Looming: a patch darker than its surroundings that gets wider on one side for two
        # steps running. Right after the fly moves itself (a jump) the whole view shifts, so
        # looming is ignored for a few steps: the efference copy a real fly uses.
        dark = self.lum < self.lum.mean() - 0.12
        if self.suppress > 0:
            self.suppress -= 1
        for s, mask in (("L", self.left), ("R", ~self.left)):
            width = int((dark & mask).sum())
            growth = 0 if first else width - self.prev_width[s]
            self.prev_width[s] = width
            self.streak[s] = self.streak[s] + 1 if growth > 0 else 0
            if self.suppress > 0 or self.streak[s] < 2:
                self.loom[s] = 0.0
            else:
                self.loom[s] = float(np.clip(growth * 0.12, 0.0, 0.5))
        return self.drive_vec


class Recorder:
    """Counts every neuron's spikes since the last snapshot."""

    def __init__(self, brain):
        self.brain = brain
        self.n = brain.n
        self.counts = np.zeros(self.n, np.int32)
        self.steps = 0
        self.since = time.time()
        self.region_of = np.full(self.n, len(REGIONS), np.int8)   # index into REGIONS; last = other
        sc = brain.superclass.astype(str) if brain.superclass is not None else np.array([""] * self.n)
        for i, (_, classes, _) in enumerate(REGIONS):
            self.region_of[np.isin(sc, classes)] = i
        self.cell_type = brain.cell_type.astype(str)
        pos = brain.positions
        self.has_pos = ~np.isnan(pos).any(axis=1) if pos is not None else np.zeros(self.n, bool)
        p = pos[self.has_pos]
        spread = np.ptp(p, axis=0)
        self.axes = np.argsort(spread)[-2:][::-1]              # the two longest axes of the nervous system
        self.xy = np.full((self.n, 2), np.nan, np.float32)
        self.xy[self.has_pos] = p[:, self.axes]
        rng = np.random.default_rng(0)
        idx = np.flatnonzero(self.has_pos)
        self.background = rng.choice(idx, min(15000, len(idx)), replace=False)

    def record(self, fired: np.ndarray):
        self.counts[fired] += 1
        self.steps += 1

    def reset(self):
        self.counts[:] = 0
        self.steps = 0
        self.since = time.time()


_worker: threading.Thread | None = None


def snapshot(out_dir: str, fly, vision: Vision, rec: Recorder, dt: float, note: str = "") -> str | None:
    """Queue one PNG (screen + what it saw + brain map) to render in the background.
    Returns the path it will be written to, or None if the previous one is still rendering."""
    global _worker
    if _worker is not None and _worker.is_alive():
        return None
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{stamp}.png")
    state = dict(
        path=path, stamp=stamp, note=note, counts=rec.counts.copy(), secs=max(rec.steps * dt, dt),
        wall=max(time.time() - rec.since, dt),
        shot=vision.shot, lum=vision.lum.copy(), drive_mean=float(vision.drive_vec.mean()),
        fov=vision.fov, reach=vision.reach,
        x=fly.x, y=fly.y, w=fly.w, h=fly.h, heading=fly.heading, food=tuple(fly.food), place=fly.place,
    )
    _worker = threading.Thread(target=_render, args=(state, rec), daemon=True)
    _worker.start()
    return path


def wait_for_snapshot(timeout: float = 15.0):
    """Block until the background render finishes (called at exit)."""
    if _worker is not None and _worker.is_alive():
        _worker.join(timeout)


def _render(st: dict, rec: Recorder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    counts, secs = st["counts"], st["secs"]
    rate = counts / secs
    in_use = rate >= IN_USE_HZ

    fig = plt.figure(figsize=(16, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[7, 2.2],
                          left=0.02, right=0.98, top=0.9, bottom=0.05, hspace=0.18, wspace=0.06)
    ax_s = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_l = fig.add_subplot(gs[1, 1])
    for ax in (ax_s, ax_p, ax_b, ax_l):
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    ax_l.axis("off")

    # --- the screen, with the fly on it ---
    if st["shot"] is not None:
        ax_s.imshow(st["shot"], interpolation="bilinear")
    ax_s.set_xlim(0, st["w"]); ax_s.set_ylim(st["h"], 0)
    hd = math.degrees(st["heading"])
    half = math.degrees(st["fov"]) / 2
    ax_s.add_patch(Wedge((st["x"], st["y"]), st["reach"], hd - half, hd + half,
                         facecolor="#3987e5", alpha=0.12, edgecolor="#3987e5", lw=1))
    ax_s.plot(*st["food"], marker="o", ms=14, mfc="none", mec="#c98500", mew=2)
    ax_s.annotate("food", st["food"], color=INK, fontsize=8, xytext=(10, -10), textcoords="offset points")
    ax_s.arrow(st["x"], st["y"], 40 * math.cos(st["heading"]), 40 * math.sin(st["heading"]),
               head_width=12, color="#e66767", lw=2, length_includes_head=True)
    ax_s.plot(st["x"], st["y"], marker="o", ms=9, color="#e66767", mec=INK, mew=1.5)
    ax_s.set_xticks([]); ax_s.set_yticks([])
    ax_s.set_title(f"what the fly sees    standing on: {st['place']}    {st['note']}",
                   color=INK, fontsize=10, loc="left")

    # --- the panorama strip ---
    strip = np.vstack([st["lum"], st["lum"]])
    ax_p.imshow(strip, cmap="gray", vmin=0, vmax=1, aspect="auto", extent=[-half, half, 0, 1])
    ax_p.set_yticks([])
    ax_p.set_xticks([-half, -half / 2, 0, half / 2, half])
    ax_p.set_xlabel("the panorama its photoreceptors got (degrees from heading); mean drive "
                    f"{st['drive_mean']:.2f}", color=INK2, fontsize=8)

    # --- the brain map ---
    xy, bg = rec.xy, rec.background
    ax_b.scatter(xy[bg, 0], xy[bg, 1], s=0.3, c="#4a4a48", linewidths=0, rasterized=True)
    idx = np.flatnonzero(in_use & rec.has_pos)
    idx = idx[np.argsort(rate[idx])]                       # busiest drawn last, on top
    colors = np.array([c for _, _, c in REGIONS] + [OTHER])
    ax_b.scatter(xy[idx, 0], xy[idx, 1], s=2 + 8 * np.log1p(rate[idx]), c=colors[rec.region_of[idx]],
                 linewidths=0, alpha=0.85, rasterized=True)
    ax_b.set_aspect("equal")
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.invert_yaxis()
    ax_b.set_title(f"brain in use, last {st['wall']:.0f} s ({secs:.1f} s of brain time): "
                   f"{int(in_use.sum()):,} of {rec.n:,} neurons above {IN_USE_HZ:.0f} Hz",
                   color=INK, fontsize=10, loc="left")
    ax_b.text(0.01, 0.01, f"brain on the left, nerve cord on the right; grey = idle; dot size = firing rate; {int(counts.sum()):,} spikes in all",
              color=INK2, transform=ax_b.transAxes, fontsize=7, va="bottom")

    # --- legend panel: region, cells in use, spikes; busiest cell types ---
    for i, (name, _, col) in enumerate(REGIONS + [("other", [], OTHER)]):
        m = rec.region_of == i
        ax_l.text(0.0, 0.97 - i * 0.105, "●", color=col, transform=ax_l.transAxes, fontsize=11, va="top")
        ax_l.text(0.035, 0.97 - i * 0.105,
                  f"{name}: {int((in_use & m).sum()):,} of {int(m.sum()):,} cells in use, {int(counts[m].sum()):,} spikes",
                  color=INK2, transform=ax_l.transAxes, fontsize=8, va="top")
    per_type: dict[str, int] = {}
    for t, c in zip(rec.cell_type[in_use], counts[in_use]):
        per_type[t] = per_type.get(t, 0) + int(c)
    top = sorted(per_type.items(), key=lambda kv: -kv[1])[:8]
    ax_l.text(0.62, 0.97, "busiest cell types (spikes)", color=INK, transform=ax_l.transAxes, fontsize=8, va="top")
    for i, (t, c) in enumerate(top):
        ax_l.text(0.62, 0.86 - i * 0.105, f"{t}  {c:,}", color=INK2, transform=ax_l.transAxes, fontsize=8, va="top")

    fig.suptitle(f"fly cursor  {st['stamp']}   position ({int(st['x'])}, {int(st['y'])})  heading {hd:.0f} deg",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.savefig(st["path"], dpi=90, facecolor=SURFACE)
    plt.close(fig)
