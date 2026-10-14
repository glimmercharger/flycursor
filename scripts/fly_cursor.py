"""fly_cursor.py - let the fruit fly connectome drive your mouse, and remember.

The mouse cursor is the fly's body. It has a heading, like a walking fly.
Senses go in through the fly's own sensory neurons, motor commands come out of
its descending neurons, and the wiring in between is the MaleCNS connectome.

  IN   LC10a   (chase target)   <- a "food" spot on screen, on the side it is on
       LPLC2   (looming)        <- the window under the cursor growing
       LC4     (threat/escape)  <- the cursor getting near a screen edge
       tpGRN   (leg taste)      <- standing on food (flies taste with their feet)
       KC      (Kenyon cells)   <- WHICH window it is standing on, as a sparse code

  OUT  DNa02 L/R (steering)     -> turn heading left or right
       DNg100    (forward)      -> walk along heading
       MDN       (backward)     -> walk backwards
       DNp01     (giant fiber)  -> jump: a leap away
       DNg11     (punch)        -> whack a key: which key depends on which descending
                                   neurons fired that step (or the next letter of --text)
       feeding                  -> reaching food = a click (unless it dislikes the place)

MEMORY. The connectome itself never changes, so the fly is given the learning
rule real flies use in the mushroom body: when dopamine arrives while a set of
Kenyon cells is active, the synapses from those Kenyon cells onto the mushroom
body output neurons (MBONs) in that compartment weaken.

  * Each window the fly stands on activates its own sparse set of Kenyon cells
    (the fly's code for "this place"), chosen from the window's title.
  * Finding food fires the PAM reward dopamine neurons and weakens that place's
    synapses onto the avoidance MBONs: the fly comes to like the place.
  * Getting scared there (a loom, a screen edge, a giant-fiber jump) fires the
    PPL1 punishment dopamine neurons and weakens the synapses onto the approach
    MBONs: the fly comes to dislike it.
  * Liked places: it chases food harder and will feed (click) there.
    Disliked places: they look threatening, so it jumps and backs away, and it
    refuses to feed there.
  * The learned synapse strengths are saved to fly_memory.json next to this
    script and re-applied to the brain next time, so it remembers across runs.

EYES. Every 2 s a screenshot becomes the fly's world: a panorama is read off
the screen around the cursor and fed to its 6,006 photoreceptors. Every
--snapshot-every seconds a PNG goes into snapshots/ showing the screen, what
it saw, and which parts of the brain fired since the last snapshot.

Safety: pyautogui's fail-safe is on, so slam the mouse into any screen corner
to abort. The run stops after --seconds. --no-click / --no-type switch those
off, and --dry-run moves nothing at all.

Usage:
  python fly_cursor.py                  # 120 s, click + type on
  python fly_cursor.py --no-type
  python fly_cursor.py --dry-run --seconds 30
  python fly_cursor.py --forget         # wipe its memory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

try:
    import pyautogui
    import pygetwindow
except ImportError:
    sys.exit("pip install pyautogui")

from flybrain import FlyBrain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fly_vision import Recorder, Vision, snapshot, wait_for_snapshot

pyautogui.FAILSAFE = True   # mouse into a corner => abort
pyautogui.PAUSE = 0         # we pace the loop ourselves
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")   # window titles can hold characters the console lacks

HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(HERE, "fly_memory.json")

# MBON compartments after Aso et al. 2014. Approach MBONs sit in compartments
# innervated by PPL1 (punishment) dopamine neurons; avoidance MBONs in PAM
# (reward) compartments. Numbering: MBON01 g5b'2a, 02 b2b'2a, 03 b'2mp, 04 b'2mp
# bilateral, 05 g4>g1g2, 06 b1>a, 07 a1, 11 g1pedc>a/b, 12 g2a'1, 14 a3.
APPROACH_MBONS = ["MBON11", "MBON14", "MBON07", "MBON06", "MBON12"]
AVOID_MBONS = ["MBON01", "MBON02", "MBON03", "MBON04", "MBON05"]
KC_TYPES = ["KCab", "KCab-p", "KCg-m", "KCg-d", "KCapbp-ap1", "KCapbp-ap2", "KCapbp-m"]
TASTE_TYPES = ["claw_tpGRN", "dorsal_tpGRN", "BM_Taste"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=300, help="how long to run (default 120)")
    ap.add_argument("--no-click", action="store_true", help="never click (feeding)")
    ap.add_argument("--no-type", action="store_true", help="never type (punch bursts)")
    ap.add_argument("--text", default=None,
                    help="if given, type these letters in order; otherwise the brain picks keys itself")
    ap.add_argument("--dry-run", action="store_true", help="print actions, move nothing")
    ap.add_argument("--speed", type=float, default=16.0, help="pixels per walking spike (default 16)")
    ap.add_argument("--turn", type=float, default=25.0, help="degrees per steering spike (default 25)")
    ap.add_argument("--food-every", type=float, default=20.0, help="seconds before the food spot moves")
    ap.add_argument("--food-radius", type=float, default=45.0, help="pixels within which it has reached food")
    ap.add_argument("--memory", default=MEMORY_FILE, help="where memories are kept")
    ap.add_argument("--forget", action="store_true", help="delete the memory file and start naive")
    ap.add_argument("--snapshot-every", type=float, default=20.0,
                    help="seconds between PNG snapshots of the screen + brain (0 = never)")
    ap.add_argument("--snapshots", default=os.path.join(HERE, "snapshots"), help="folder for the PNGs")
    ap.add_argument("--no-eyes", action="store_true", help="do not feed the screen into the photoreceptors")
    ap.add_argument("--seed", type=int, default=64)
    return ap.parse_args()


def place_name(x: int, y: int) -> str:
    """What the fly is standing on: the app behind the window under the cursor."""
    try:
        for w in pygetwindow.getWindowsAt(x, y):
            t = (w.title or "").strip()
            if t and w.visible and w.width > 50 and w.height > 50:
                # "doc.txt - Notepad" -> "Notepad", so memories group per app
                t = t.encode("ascii", "ignore").decode()          # spinners and emoji change; drop them
                t = t.rsplit(" - ", 1)[-1].rsplit(" | ", 1)[-1]
                t = " ".join(t.split()).strip(" -|?:.")[:40]
                if t:
                    return t
    except Exception:
        pass
    return "desktop"


class MushroomBody:
    """Dopamine-gated plasticity on Kenyon cell -> MBON synapses, kept per place."""

    LEARN = 0.35      # fraction of remaining strength lost per dopamine event
    FLOOR = 0.05      # synapses never go fully silent
    DRIVE = 0.45      # voltage per step into a place's Kenyon cells
    ENSEMBLE = 150    # Kenyon cells per place (~4% of 4,064, like a real odour)

    def __init__(self, brain: FlyBrain, path: str, forget: bool):
        self.brain = brain
        self.path = path
        self.kc = brain.cells(KC_TYPES)
        self.approach = brain.cells(APPROACH_MBONS)
        self.avoid = brain.cells(AVOID_MBONS)
        self.pam = brain.cells([f"PAM{i:02d}" for i in range(1, 16)])
        self.ppl1 = brain.cells([f"PPL1{i:02d}" for i in range(1, 9)])
        self.taste = brain.cells(TASTE_TYPES)
        self.base = brain.weights.copy()      # naive synapse strengths
        post = brain.indices
        self.is_app = np.zeros(brain.n, bool); self.is_app[self.approach] = True
        self.is_avoid = np.zeros(brain.n, bool); self.is_avoid[self.avoid] = True
        self.post_app, self.post_avoid = self.is_app[post], self.is_avoid[post]
        self.places: dict[str, dict] = {}
        self.ensembles: dict[str, np.ndarray] = {}
        self.edges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.dopamine: list = []              # (cells, voltage, steps left)
        if forget and os.path.exists(path):
            os.remove(path)
        if os.path.exists(path):
            with open(path) as f:
                self.places = json.load(f)
            for name in self.places:
                self.apply(name)

    # -- a place's code and synapses --
    def ensemble(self, name: str) -> np.ndarray:
        if name not in self.ensembles:
            seed = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            self.ensembles[name] = np.sort(rng.choice(self.kc, self.ENSEMBLE, replace=False))
            ip = self.brain.indptr
            e = np.concatenate([np.arange(ip[j], ip[j + 1]) for j in self.ensembles[name]])
            self.edges[name] = (e[self.post_app[e]], e[self.post_avoid[e]])
        return self.ensembles[name]

    def record(self, name: str) -> dict:
        self.ensemble(name)
        return self.places.setdefault(name, {"approach": 1.0, "avoid": 1.0, "seen_s": 0.0,
                                             "rewards": 0, "punishments": 0, "last_seen": ""})

    def apply(self, name: str):
        """Write a place's learned synapse strengths into the brain."""
        r = self.record(name)
        e_app, e_avoid = self.edges[name]
        self.brain.weights[e_app] = self.base[e_app] * r["approach"]
        self.brain.weights[e_avoid] = self.base[e_avoid] * r["avoid"]

    def valence(self, name: str) -> float:
        """+ = likes the place, - = dislikes it. Read straight off the synapses."""
        r = self.record(name)
        return r["approach"] - r["avoid"]

    # -- the learning rule --
    def reward(self, name: str):
        r = self.record(name)
        r["avoid"] = max(self.FLOOR, r["avoid"] * (1 - self.LEARN))
        r["rewards"] += 1
        self.apply(name)
        self.dopamine.append([self.pam, 0.5, 10])

    def punish(self, name: str):
        r = self.record(name)
        r["approach"] = max(self.FLOOR, r["approach"] * (1 - self.LEARN))
        r["punishments"] += 1
        self.apply(name)
        self.dopamine.append([self.ppl1, 0.8, 10])

    # -- per step --
    def inject(self, name: str, dt: float) -> list:
        r = self.record(name)
        r["seen_s"] += dt
        r["last_seen"] = datetime.now().isoformat(timespec="seconds")
        out = [(self.ensemble(name), self.DRIVE)]
        for d in self.dopamine:
            out.append((d[0], d[1]))
            d[2] -= 1
        self.dopamine = [d for d in self.dopamine if d[2] > 0]
        return out

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.places, f, indent=1)

    def table(self) -> str:
        rows = sorted(self.places.items(), key=lambda kv: -kv[1]["seen_s"])
        lines = [f"{'place':40} {'feels':>6} {'seen':>6} {'food':>4} {'scares':>6}"]
        for name, r in rows:
            v = r["approach"] - r["avoid"]
            feel = "likes" if v > 0.15 else "hates" if v < -0.15 else "meh"
            lines.append(f"{name:40} {feel:>6} {r['seen_s']:5.0f}s {r['rewards']:4d} {r['punishments']:6d}")
        return "\n".join(lines)


class Fly:
    """The cursor as a fly body: a position and a heading."""

    def __init__(self, brain: FlyBrain, mb: MushroomBody, args):
        self.brain, self.mb, self.args = brain, mb, args
        self.w, self.h = pyautogui.size()
        x, y = pyautogui.position()
        self.x, self.y = float(x), float(y)
        self.heading = random.uniform(0, 2 * math.pi)
        self.keys = "abcdefghijklmnopqrstuvwxyz0123456789      "   # the keyboard it whacks
        self.food = self.new_food()
        self.food_at = time.time()
        self.prev_area = None
        self.prev_area_place = None
        self.last_scare = 0.0
        self.text_i = 0
        self.margin = 40
        self.place = place_name(int(self.x), int(self.y))
        self.place_checked = 0
        self.vision = None                                # set by main when the eyes are on
        self.events: list[str] = []

        c = brain.cells
        self.inp = {
            "chaseL": c(["LC10a"], "L"), "chaseR": c(["LC10a"], "R"),
            "loomL": c(["LPLC2"], "L"), "loomR": c(["LPLC2"], "R"),
            "threatL": c(["LC4"], "L"), "threatR": c(["LC4"], "R"),
        }
        self.out = {
            "steerL": c(["DNa02"], "L"), "steerR": c(["DNa02"], "R"),
            "forward": c(["DNg100"]), "backward": c(["MDN"]),
            "jump": c(["DNp01"]), "punch": c(["DNg11"]),
            "dn": c(["descending_neuron"]),
            "mbon+": mb.approach, "mbon-": mb.avoid,
        }
        self.counts = {k: 0 for k in self.out}
        self.did = {"move": 0, "jump": 0, "click": 0, "type": 0, "fed": 0, "scared": 0}

    def new_food(self):
        m = 120
        return (random.uniform(m, self.w - m), random.uniform(m, self.h - m))

    # ---- senses --------------------------------------------------------
    def sense(self) -> list:
        """Build (neuron indices, voltage) pairs for this step."""
        a, mb = self.args, self.mb
        inject = []
        self.place_checked += 1
        if self.place_checked % 10 == 0:                   # 5 Hz is plenty for EnumWindows
            self.place = place_name(int(self.x), int(self.y))
        v = mb.valence(self.place)
        inject += mb.inject(self.place, self.brain.dt)

        # Food on the left or right of the heading drives the chase neurons on that side.
        if time.time() - self.food_at > a.food_every:
            self.food, self.food_at = self.new_food(), time.time()
        dx, dy = self.food[0] - self.x, self.food[1] - self.y
        dist = math.hypot(dx, dy)
        if dist < a.food_radius:
            self.feed()
            dx, dy = self.food[0] - self.x, self.food[1] - self.y
            dist = math.hypot(dx, dy)
        rel = math.atan2(dy, dx) - self.heading          # angle of food relative to heading
        rel = (rel + math.pi) % (2 * math.pi) - math.pi  # -pi..pi, negative = left
        side = "L" if rel < 0 else "R"
        angle = 40.0 / max(dist, 8.0)                     # apparent size of the food spot
        chase = 0.6 + 0.2 * angle + 0.3 * min(abs(rel) / math.pi, 1.0)
        chase *= 1 + max(v, 0.0)                          # a liked place: keener on food
        inject.append((self.inp["chase" + side], float(np.clip(chase, 0, 0.8))))

        # Something dark growing wider in its field of view is looming (from the eyes).
        if self.vision is not None:
            for s in "LR":
                if self.vision.loom[s] > 0:
                    inject.append((self.inp["loom" + s], self.vision.loom[s]))
                    if self.vision.loom[s] >= 0.45:
                        self.scare("saw something loom")
        # The window under the cursor getting bigger looks like something looming.
        try:
            wins = pygetwindow.getWindowsAt(int(self.x), int(self.y))
            area = (wins[0].width * wins[0].height) if wins else 0
        except Exception:
            area = 0
        if self.place != self.prev_area_place:        # a different window is just a different place
            self.prev_area, self.prev_area_place = None, self.place
        if self.prev_area is not None and area > self.prev_area * 1.05:
            growth = min((area - self.prev_area) / (self.w * self.h), 1.0)
            loom = float(np.clip(growth * 10.0, 0.2, 0.8))
            inject.append((self.inp["loomL"], loom))
            inject.append((self.inp["loomR"], loom))
            self.scare("loom")
        self.prev_area = area

        # Near a screen edge is a threat on that side. A disliked place is a threat all round.
        m = self.margin * 3
        threat_l = max(0.0, 1 - self.x / m) + max(-v, 0.0) * 0.15
        threat_r = max(0.0, 1 - (self.w - self.x) / m) + max(-v, 0.0) * 0.15
        if threat_l > 0:
            inject.append((self.inp["threatL"], float(min(0.8 * threat_l, 0.8))))
        if threat_r > 0:
            inject.append((self.inp["threatR"], float(min(0.8 * threat_r, 0.8))))
        return inject

    def feed(self):
        """Standing on food: taste it with the feet, get the reward, and (click) eat."""
        mb, a = self.mb, self.args
        self.did["fed"] += 1
        mb.dopamine.append([mb.taste, 0.6, 5])
        mb.reward(self.place)
        self.events.append(f"FED in {self.place}")
        if not a.no_click and mb.valence(self.place) >= -0.15:
            self.did["click"] += 1
            if a.dry_run:
                print(f"  CLICK (feeding) at ({int(self.x)}, {int(self.y)}) in {self.place}")
            else:
                pyautogui.click()
        self.food, self.food_at = self.new_food(), time.time()

    def scare(self, why: str):
        if time.time() - self.last_scare < 1.0:        # one fright = one lesson
            return
        self.last_scare = time.time()
        self.did["scared"] += 1
        self.mb.punish(self.place)
        self.events.append(f"SCARED ({why}) in {self.place}")

    # ---- motor ---------------------------------------------------------
    def act(self, fired: np.ndarray):
        a = self.args
        n = {k: int(np.isin(fired, idx).sum()) for k, idx in self.out.items()}
        for k, v in n.items():
            self.counts[k] += v

        turn = (n["steerR"] - n["steerL"]) * math.radians(a.turn)
        self.heading = (self.heading + turn) % (2 * math.pi)
        # DNg100 (forward walking) almost never fires in this model, as the author also
        # found, so a steering spike carries half a step too: the fly turns while walking.
        step = (n["forward"] + 0.5 * (n["steerL"] + n["steerR"]) - n["backward"]) * a.speed
        if n["jump"]:
            step += 80 * n["jump"]
            self.did["jump"] += 1
            if self.vision is not None:
                self.vision.self_moved()                  # my own leap is not something looming
        if step:
            self.x += math.cos(self.heading) * step
            self.y += math.sin(self.heading) * step
            # bounce off the edges, like hitting the wall of a dish
            hit = False
            if not (self.margin <= self.x <= self.w - self.margin):
                self.heading, hit = math.pi - self.heading, True
            if not (self.margin <= self.y <= self.h - self.margin):
                self.heading, hit = -self.heading, True
            if hit:
                self.scare("edge")
            self.x = float(np.clip(self.x, self.margin, self.w - self.margin))
            self.y = float(np.clip(self.y, self.margin, self.h - self.margin))
            self.did["move"] += 1
            if not a.dry_run:
                pyautogui.moveTo(int(self.x), int(self.y))

        # The six DNg11 cells tick at ~6 Hz on their own, so only a burst (two or more
        # in the same 20 ms step) counts as a punch.
        if n["punch"] >= 2 and not a.no_type:
            if a.text:
                ch = a.text[self.text_i % len(a.text)]
            else:
                # which key gets hit depends on exactly which descending neurons fired this step
                dn = fired[np.isin(fired, self.out["dn"])]
                ch = self.keys[int(dn.sum() + len(dn) * 7919) % len(self.keys)]
            self.text_i += 1
            self.did["type"] += 1
            if a.dry_run:
                print(f"  TYPE {ch!r}")
            else:
                pyautogui.write(ch)
        return n


def main():
    args = parse()
    print("loading the fly brain (166,700 neurons)...", flush=True)
    # brain files live in fly-data next to this folder (Downloads/fly-data); falls back to ~/fly-data
    data = os.path.join(os.path.dirname(HERE), "fly-data")
    brain = FlyBrain(device="cpu", seed=args.seed, data=data if os.path.isdir(data) else None,
                     sensory_input=False)   # otherwise olfactory neurons run away at rest
    mb = MushroomBody(brain, args.memory, args.forget)
    fly = Fly(brain, mb, args)
    vision, rec = Vision(brain), Recorder(brain)
    if not args.no_eyes:
        vision.grab()
        fly.vision = vision
    brain.step()  # warms up the numba kernel
    print(f"screen {fly.w}x{fly.h}; food at {tuple(int(v) for v in fly.food)}; "
          f"click={'off' if args.no_click else 'on'} type={'off' if args.no_type else 'on'} "
          f"dry_run={args.dry_run}")
    if mb.places:
        print(f"remembers {len(mb.places)} places from {args.memory}:\n{mb.table()}")
    else:
        print("no memories yet")
    if args.snapshot_every > 0:
        print(f"snapshots every {args.snapshot_every:.0f} s -> {args.snapshots}")
    print("move the mouse into a screen corner to abort\n")

    dt = brain.dt
    t_end = time.time() + args.seconds
    last_report = last_save = last_snap = time.time()
    window = {k: 0 for k in fly.out if k != "dn"}
    try:
        while time.time() < t_end:
            t0 = time.time()
            eye = None
            if not args.no_eyes:
                if time.time() - vision.grabbed_at > 2.0:     # a fresh look at the screen every 2 s
                    vision.grab()
                eye = vision.drive(fly.x, fly.y, fly.heading)
            fired = brain.step(eye_drive=eye, inject=fly.sense())
            rec.record(fired)
            n = fly.act(fired)
            for k in window:
                window[k] += n[k]
            if time.time() - last_report >= 1.0:
                last_report = time.time()
                for ev in fly.events:
                    print(f"  {ev}")
                fly.events.clear()
                rates = " ".join(f"{k}={v:2d}" for k, v in window.items())
                print(f"t={args.seconds - (t_end - time.time()):5.1f}s pos=({int(fly.x):4d},{int(fly.y):4d}) "
                      f"on={fly.place[:18]:18} feels={mb.valence(fly.place):+.2f}  {rates}", flush=True)
                window = {k: 0 for k in fly.out if k != "dn"}
            if args.snapshot_every > 0 and time.time() - last_snap >= args.snapshot_every:
                last_snap = time.time()
                vision.grab()
                path = snapshot(args.snapshots, fly, vision, rec, dt, note=f"feels {mb.valence(fly.place):+.2f}")
                if path:
                    rec.reset()
                    print(f"  SNAPSHOT -> {path}", flush=True)
            if time.time() - last_save >= 10.0:
                last_save = time.time()
                mb.save()
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except pyautogui.FailSafeException:
        print("\nfail-safe: mouse hit a corner, stopping")
    except KeyboardInterrupt:
        print("\nstopped")
    mb.save()
    wait_for_snapshot()
    print("\ntotals:", fly.did)
    print("spikes:", fly.counts)
    print(f"\nmemory ({args.memory}):\n{mb.table()}")


if __name__ == "__main__":
    main()
