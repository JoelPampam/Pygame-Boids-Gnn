#!/usr/bin/env python3
"""
run_experiment.py

Randomizes the Boids simulation's tunable parameters (within sensible
ranges) and runs main.py headlessly to generate a dataset.

Folder layout produced under Data/:

    Data/
      dataset_1/
        config.txt          <- human-readable params for this dataset
        config.json          <- same params, used internally to reload them
        boids_log_1.csv / .bin   <- one generation run under that config
        boids_log_2.csv / .bin   <- another run, same config, new seed
      dataset_2/              <- a brand new randomized config
        ...

Usage:
    python run_experiment.py
        Pick a fresh random config, create the next Data/dataset_N/
        folder, and generate boids_log_1.csv/.bin inside it.

    python run_experiment.py --add Data/dataset_3
        Reuse dataset_3's exact stored config and generate the next
        numbered run (e.g. boids_log_2.csv/.bin) inside it, with a
        new random seed.

    python run_experiment.py --add Data/dataset_3 --seed 12345
        Same as above, but pin the seed -- use this to exactly
        regenerate one specific numbered run.

    python run_experiment.py --steps 10000
        Override how many steps get recorded (default 37500).

    python run_experiment.py --show
        Open the pygame window and watch it run instead of headless.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time

#run Python Experiments.py --add Data/dataset_# to run the same configuration again
#run Python Experiments.py --set Name=value to give it a specific constraint
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(PROJECT_ROOT, "Pygame-Boids-GNN.py")
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

#About 1 minute
DEFAULT_STEPS = 3600


# Ranges are hand-picked around the defaults already in config.py so
# randomized runs stay "sensible" instead of producing a degenerate sim
# (e.g. boids that never see each other, or a predator that can never
# catch anything).
FLOAT_RANGES = {
    "BOIDS_SEPARATION_WEIGHT": (0.01, 0.05),
    "BOIDS_COHESION_WEIGHT": (0.0005, 0.003),
    "BOIDS_ALIGNMENT_WEIGHT": (0.02, 0.08),
    "BOIDS_MIN_SPEED": (0.5, 2.0),
    "BOIDS_LEADER_FOLLOW_WEIGHT": (0.01, 0.05),
    "BOIDS_LEADER_WANDER_STRENGTH": (0.05, 0.3),
    "BOIDS_LEADER_WANDER_FORCE": (0.05, 0.2),
    "BOIDS_WAYPOINT_WEIGHT": (0.005, 0.02),
    "BOIDS_PREDATOR_AVOID_WEIGHT": (1.5, 5.0),
    "BOIDS_PREDATOR_CHASE_WEIGHT": (0.005, 0.02),
    "BOIDS_PREDATOR_MIN_SPEED": (1.0, 2.5),
    "BOIDS_BORDER_TURN_WEIGHT": (1.0, 2.5),
    "BOIDS_BORDER_BOUNCE_KICK": (0.5, 2.0),
    "BOIDS_BORDER_EDGE_DRAG": (0.2, 0.6),
}

INT_RANGES = {
    "BOIDS_SEPARATION_RADIUS": (20, 50),
    "BOIDS_LEADER_FOLLOW_RADIUS": (150, 350),
    "BOIDS_WAYPOINT_RADIUS": (20, 60),
    "BOIDS_NUM_LEADERS": (1, 5),
    "BOIDS_NUM_PREDATORS": (0, 3),
    "BOIDS_PREDATOR_AVOID_RADIUS": (80, 180),
    "BOIDS_BORDER_MARGIN": (60, 150),
}

CHOICE_RANGES = {
    "BOIDS_BORDER_MODE": ["wrap", "bounded"],
    "BOIDS_WAYPOINT_LOOP": [True, False],
}


# Derived parameters aren't sampled from a fixed range -- they're computed
# relative to another sampled value (see sample_config below). They can
# still be pinned exactly with --set, just not narrowed with --range.
DERIVED_FLOAT_PARAMS = {"BOIDS_MAX_SPEED", "BOIDS_PREDATOR_MAX_SPEED"}
DERIVED_INT_PARAMS = {"BOIDS_COHESION_RADIUS", "BOIDS_ALIGNMENT_RADIUS"}

ALL_FLOAT_PARAMS = set(FLOAT_RANGES) | DERIVED_FLOAT_PARAMS
ALL_INT_PARAMS = set(INT_RANGES) | DERIVED_INT_PARAMS
ALL_PARAM_NAMES = ALL_FLOAT_PARAMS | ALL_INT_PARAMS | set(CHOICE_RANGES)


def normalize_param_name(name):
    name = name.strip().upper()
    if not name.startswith("BOIDS_"):
        name = "BOIDS_" + name
    return name


def coerce_set_value(name, raw):
    """Turn a --set NAME=VALUE string into the right Python type for NAME."""
    if name in ALL_FLOAT_PARAMS:
        return float(raw)
    if name in ALL_INT_PARAMS:
        return int(float(raw))
    if name == "BOIDS_BORDER_MODE":
        if raw not in ("wrap", "bounded"):
            sys.exit(f"--set BORDER_MODE must be 'wrap' or 'bounded', got: {raw}")
        return raw
    if name == "BOIDS_WAYPOINT_LOOP":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    sys.exit(f"Unknown parameter for --set: {name}\nKnown parameters: "
              f"{', '.join(sorted(p.replace('BOIDS_', '') for p in ALL_PARAM_NAMES))}")


def parse_set_args(pairs):
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"--set expects NAME=VALUE, got: {pair}")
        name, raw = pair.split("=", 1)
        name = normalize_param_name(name)
        overrides[name] = coerce_set_value(name, raw)
    return overrides


def parse_range_args(pairs):
    """--range only applies to independently-sampled params (FLOAT_RANGES /
    INT_RANGES) -- not the derived ones, and not the choice params (use
    --set for those instead)."""
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"--range expects NAME=LOW,HIGH, got: {pair}")
        name, rng = pair.split("=", 1)
        name = normalize_param_name(name)
        if "," not in rng:
            sys.exit(f"--range expects NAME=LOW,HIGH, got: {pair}")
        lo, hi = rng.split(",", 1)
        if name in FLOAT_RANGES:
            overrides[name] = (float(lo), float(hi))
        elif name in INT_RANGES:
            overrides[name] = (int(float(lo)), int(float(hi)))
        else:
            sys.exit(
                f"--range doesn't apply to {name.replace('BOIDS_', '')} "
                f"(either it's a derived value -- use --set to pin it -- or "
                f"a wrap/bounded-style choice -- use --set for that too).\n"
                f"Rangeable parameters: "
                f"{', '.join(sorted(p.replace('BOIDS_', '') for p in (set(FLOAT_RANGES) | set(INT_RANGES))))}"
            )
    return overrides


def sample_config(range_overrides=None, set_overrides=None):
    """Sample every randomizable parameter, respecting the dependent
    relationships the simulation actually relies on. `range_overrides`
    narrows/widens the sampling bounds for specific params; `set_overrides`
    pins params to an exact value afterward (skipping randomization for
    those entirely)."""
    range_overrides = range_overrides or {}
    set_overrides = set_overrides or {}
    cfg = {}

    float_ranges = {**FLOAT_RANGES, **{k: v for k, v in range_overrides.items() if k in FLOAT_RANGES}}
    int_ranges = {**INT_RANGES, **{k: v for k, v in range_overrides.items() if k in INT_RANGES}}

    for name, (lo, hi) in float_ranges.items():
        cfg[name] = round(random.uniform(lo, hi), 5)
    for name, (lo, hi) in int_ranges.items():
        cfg[name] = random.randint(lo, hi)
    for name, choices in CHOICE_RANGES.items():
        cfg[name] = random.choice(choices)

    # Radii need SEPARATION <= COHESION <= ALIGNMENT, or a boid would try
    # to align with neighbors it hasn't even cohered toward yet.
    sep = cfg["BOIDS_SEPARATION_RADIUS"]
    cfg["BOIDS_COHESION_RADIUS"] = random.randint(sep, sep + 40)
    coh = cfg["BOIDS_COHESION_RADIUS"]
    cfg["BOIDS_ALIGNMENT_RADIUS"] = random.randint(coh + 10, coh + 90)

    # MAX_SPEED must clear MIN_SPEED by a comfortable margin.
    min_speed = cfg["BOIDS_MIN_SPEED"]
    cfg["BOIDS_MAX_SPEED"] = round(random.uniform(min_speed + 1.5, min_speed + 4.0), 3)

    pred_min = cfg["BOIDS_PREDATOR_MIN_SPEED"]
    cfg["BOIDS_PREDATOR_MAX_SPEED"] = round(random.uniform(pred_min + 1.0, pred_min + 3.0), 3)

    # Pins applied last, so a --set value always wins regardless of any
    # relationship logic above (e.g. --set MAX_SPEED=4 overrides whatever
    # the min-speed-relative sampling above would have produced).
    for name, value in set_overrides.items():
        if name not in ALL_PARAM_NAMES:
            sys.exit(f"Unknown parameter for --set: {name}")
        cfg[name] = value

    return cfg


def next_dataset_dir():
    """Data/dataset_1, Data/dataset_2, ... -- next unused number."""
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = []
    for name in os.listdir(DATA_DIR):
        m = re.fullmatch(r"dataset_(\d+)", name)
        if m:
            existing.append(int(m.group(1)))
    n = max(existing, default=0) + 1
    return os.path.join(DATA_DIR, f"dataset_{n}")


def next_run_number(dataset_dir):
    """boids_log_1.csv, boids_log_2.csv, ... -- next unused number in an
    existing dataset folder."""
    existing = []
    if os.path.isdir(dataset_dir):
        for name in os.listdir(dataset_dir):
            m = re.fullmatch(r"boids_log_(\d+)\.csv", name)
            if m:
                existing.append(int(m.group(1)))
    return max(existing, default=0) + 1


def write_config_files(dataset_dir, cfg):
    """Write config.json (used internally by --add) and config.txt (for
    you to read). Called once, when the dataset folder is first created."""
    manifest = {"config": cfg, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    with open(os.path.join(dataset_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _write_config_txt(dataset_dir, manifest)


def _write_config_txt(dataset_dir, manifest):
    cfg = manifest["config"]
    lines = [
        f"Boids dataset: {os.path.basename(dataset_dir)}",
        f"Created:       {manifest['created']}",
        "",
        "Parameters:",
    ]
    for k in sorted(cfg):
        lines.append(f"  {k.replace('BOIDS_', ''):<28} {cfg[k]}")
    lines.append("")
    lines.append("Runs:")
    for run in manifest["runs"]:
        lines.append(
            f"  boids_log_{run['run']}.csv/.bin   "
            f"seed={run['seed']}   steps={run['steps']}   {run['timestamp']}"
        )
    with open(os.path.join(dataset_dir, "config.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def record_run(dataset_dir, run_number, seed, steps):
    """Append this run to config.json['runs'] and regenerate config.txt."""
    manifest_path = os.path.join(dataset_dir, "config.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["runs"].append({
        "run": run_number,
        "seed": seed,
        "steps": steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _write_config_txt(dataset_dir, manifest)


def run_simulation(dataset_dir, run_number, cfg, seed, steps, headless):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in cfg.items()})
    env["BOIDS_OUTPUT_DIR"] = dataset_dir
    env["BOIDS_CSV_NAME"] = f"boids_log_{run_number}.csv"
    env["BOIDS_BIN_NAME"] = f"boids_log_{run_number}.bin"
    env["BOIDS_RANDOM_SEED"] = str(seed)
    env["BOIDS_RECORD_STEPS"] = str(steps)
    env["BOIDS_HEADLESS_MODE"] = "0" if headless is False else "1"
    if headless:
        # avoids needing a real display to init pygame in headless mode
        env.setdefault("SDL_VIDEODRIVER", "dummy")

    print(f"Running boids_log_{run_number} in {dataset_dir} "
          f"(seed={seed}, steps={steps}, headless={headless})")
    subprocess.run([sys.executable, MAIN_PY], env=env, check=True, cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--add", metavar="DATASET_DIR", default=None,
                         help="Add another run to an existing Data/dataset_N folder "
                              "instead of creating a new randomized config")
    parser.add_argument("--seed", type=int, default=None,
                         help="Pin the random seed (default: random each run)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                         help=f"Steps to record (default: {DEFAULT_STEPS})")
    parser.add_argument("--show", action="store_true",
                         help="Open the pygame window instead of running headless")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                         help="Pin a parameter to an exact value instead of randomizing it "
                              "(repeatable). Example: --set NUM_PREDATORS=2 --set BORDER_MODE=wrap")
    parser.add_argument("--range", dest="ranges", action="append", default=[], metavar="NAME=LOW,HIGH",
                         help="Narrow/widen the sampling range for a parameter (repeatable, "
                              "only for new datasets, not --add). "
                              "Example: --range COHESION_WEIGHT=0.001,0.0015")
    args = parser.parse_args()

    if args.add and (args.set or args.ranges):
        sys.exit("--set/--range only apply when creating a new dataset, not with --add "
                  "(--add always reuses the dataset's existing stored config).")

    if args.add:
        dataset_dir = os.path.abspath(args.add)
        manifest_path = os.path.join(dataset_dir, "config.json")
        if not os.path.isfile(manifest_path):
            sys.exit(f"No config.json found in {dataset_dir} -- is that a dataset folder?")
        with open(manifest_path) as f:
            cfg = json.load(f)["config"]
        run_number = next_run_number(dataset_dir)
    else:
        set_overrides = parse_set_args(args.set)
        range_overrides = parse_range_args(args.ranges)
        cfg = sample_config(range_overrides=range_overrides, set_overrides=set_overrides)
        dataset_dir = next_dataset_dir()
        os.makedirs(dataset_dir, exist_ok=True)
        write_config_files(dataset_dir, cfg)
        run_number = 1

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    run_simulation(dataset_dir, run_number, cfg, seed, args.steps, headless=not args.show)
    record_run(dataset_dir, run_number, seed, args.steps)

    print(f"Done. boids_log_{run_number}.csv/.bin + config.txt in: {dataset_dir}")


if __name__ == "__main__":
    main()