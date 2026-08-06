import os

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
# Every simulation step (i.e. every unpaused frame), each boid's kinematic
# state and its "adjacency" info (which other boids/predators fall within
# each behavior radius) is written as one row to a CSV file, and as one
# packed record to a binary file. Predators get their own rows too
# (entity_type == "predator").
#
# Radii used to decide adjacency -- these match the distances already used
# inside Boid.separation / Boid.cohesion / Boid.alignment below.

# ---------------------------------------------------------------------------
# Env-var override helpers
# ---------------------------------------------------------------------------
# Every tunable parameter below can be overridden by setting the matching
# BOIDS_<NAME> environment variable. This lets run_experiment.py randomize a
# run (or replay a stored one exactly) by launching main.py as a subprocess
# with these set, without ever touching this file. Running main.py directly,
# with none of these set, behaves exactly as before -- every default here is
# unchanged from the original values.


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_int(name, default):
    v = os.environ.get(name)
    return int(float(v)) if v is not None else default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name, default):
    return os.environ.get(name, default)


# Random seed for this run's simulation (boid spawn positions/velocities,
# leader wander, etc). None -> Python's own default (unseeded) behavior,
# same as before this feature existed. run_experiment.py always sets this,
# so every generated dataset file is exactly reproducible.
_seed_env = os.environ.get("BOIDS_RANDOM_SEED")
RANDOM_SEED = int(_seed_env) if _seed_env is not None else None

# ==========================================================
# Boids Behavior
# ==========================================================
SEPARATION_RADIUS = _env_int("BOIDS_SEPARATION_RADIUS", 35)
COHESION_RADIUS = _env_int("BOIDS_COHESION_RADIUS", 50)
ALIGNMENT_RADIUS = _env_int("BOIDS_ALIGNMENT_RADIUS", 100)
# Force weights -- how strongly each rule pulls on a boid's velocity.
SEPARATION_WEIGHT = _env_float("BOIDS_SEPARATION_WEIGHT", 0.02)
COHESION_WEIGHT = _env_float("BOIDS_COHESION_WEIGHT", 0.001)
ALIGNMENT_WEIGHT = _env_float("BOIDS_ALIGNMENT_WEIGHT", 0.05)
MIN_SPEED = _env_float("BOIDS_MIN_SPEED", 1)
MAX_SPEED = _env_float("BOIDS_MAX_SPEED", 4)
# ==========================================================
# Leader / Follower
# ==========================================================
NUM_LEADERS = _env_int("BOIDS_NUM_LEADERS", 3)                 # first N boids created are designated leaders
LEADER_FOLLOW_RADIUS = _env_int("BOIDS_LEADER_FOLLOW_RADIUS", 250)      # followers only react to a leader within this range
LEADER_FOLLOW_WEIGHT = _env_float("BOIDS_LEADER_FOLLOW_WEIGHT", 0.03)     # how strongly a follower steers toward its nearest leader
LEADER_WANDER_STRENGTH = _env_float("BOIDS_LEADER_WANDER_STRENGTH", 0.15)   # how sharply a leader's own wander heading can turn per step
LEADER_WANDER_FORCE = _env_float("BOIDS_LEADER_WANDER_FORCE", 0.12)      # how strongly the wander heading pulls a leader's velocity
LEADER_COLOR = (255, 210, 60)
FOLLOWER_COLOR = (100, 200, 255)
# ==========================================================
# Waypoints
# ==========================================================
# Numbered goal spots that leaders navigate through, in order. Followers
# aren't pulled toward waypoints directly -- they just follow the nearest
# leader as before, so the whole flock ends up tracing the route as the
# leaders lead it there. Each leader tracks its own progress along this
# list independently (self.waypoint_index), so leaders don't have to be
# in lockstep with each other.
WAYPOINTS = [
    (150, 100),
    (750, 120),
    (750, 550),
    (150, 550),
    (450, 325),
]
WAYPOINT_RADIUS = _env_int("BOIDS_WAYPOINT_RADIUS", 40)         # how close a leader must get to count as "arrived"
WAYPOINT_WEIGHT = _env_float("BOIDS_WAYPOINT_WEIGHT", 0.01)       # how strongly a leader steers toward its current waypoint
WAYPOINT_LOOP = _env_bool("BOIDS_WAYPOINT_LOOP", True)         # once the last waypoint is reached, start over at 0; if
                              # False, a leader just holds at the final waypoint's id
WAYPOINT_COLOR = (120, 230, 140)
WAYPOINT_RING_COLOR = (70, 140, 85)
# ==========================================================
# Predator
# ==========================================================
NUM_PREDATORS = _env_int("BOIDS_NUM_PREDATORS", 3)
PREDATOR_AVOID_RADIUS = _env_int("BOIDS_PREDATOR_AVOID_RADIUS", 120)     # boids inside this radius of a predator will flee
PREDATOR_AVOID_WEIGHT = _env_float("BOIDS_PREDATOR_AVOID_WEIGHT", 3.0)     # strength of the flee force
PREDATOR_CHASE_WEIGHT = _env_float("BOIDS_PREDATOR_CHASE_WEIGHT", 0.01)    # how strongly a predator steers toward the nearest boid
PREDATOR_MIN_SPEED = _env_float("BOIDS_PREDATOR_MIN_SPEED", 1.5)
PREDATOR_MAX_SPEED = _env_float("BOIDS_PREDATOR_MAX_SPEED", 3.5)
PREDATOR_COLOR = (220, 40, 40)
# ---------------------------------------------------------------------------
# World border settings
# ---------------------------------------------------------------------------
# "wrap"    -> classic torus/"sphere" behavior: exiting one edge re-enters on
#              the opposite edge (this was the only behavior before).
# "bounded" -> "end of the world": boids/predators cannot cross the edge --
#              they get steered back in and lose their along-wall speed as
#              they approach it, then hard-bounced so they never actually
#              leave the screen or glide along an edge indefinitely.
# Press B during the simulation to toggle this live; it also changes what
# gets written to the log files (see the "border_mode" column/field).
BORDER_MODE = _env_str("BOIDS_BORDER_MODE", "wrap")            # "wrap" or "bounded"
BORDER_MARGIN = _env_int("BOIDS_BORDER_MARGIN", 100)             # bounded mode: distance from edge where turning starts
BORDER_TURN_WEIGHT = _env_float("BOIDS_BORDER_TURN_WEIGHT", 1.6)        # bounded mode: how hard to steer away from the edge -- has
                                 # to be bigger than it looks like it should, since it needs
                                 # to be able to out-muscle a whole cluster of boids all
                                 # cohering toward each other right at the wall
BORDER_BOUNCE_KICK = _env_float("BOIDS_BORDER_BOUNCE_KICK", 1.0)        # bounded mode: minimum outward speed added on contact
                                 # with the wall itself -- prevents stalling in a corner
BORDER_EDGE_DRAG = _env_float("BOIDS_BORDER_EDGE_DRAG", 0.4)          # bounded mode: how much along-wall (tangential) speed
                                 # bleeds off near a wall -- prevents gliding along the edge

WORLD_WIDTH = 900
WORLD_HEIGHT = 650

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the logs go, and what they're named. run_experiment.py points
# OUTPUT_DIR at Data/dataset_N/ and names each generation boids_log_<run>.*
# via these two env vars; running main.py directly (none of this set) keeps
# the original behavior of writing boids_log.csv/.bin next to config.py.
OUTPUT_DIR = os.environ.get("BOIDS_OUTPUT_DIR", SCRIPT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_NAME = os.environ.get("BOIDS_CSV_NAME", "boids_log.csv")
BIN_NAME = os.environ.get("BOIDS_BIN_NAME", "boids_log.bin")

CSV_PATH = os.path.join(OUTPUT_DIR, CSV_NAME)
BIN_PATH = os.path.join(OUTPUT_DIR, BIN_NAME)
OBSTACLE_RADIUS = 40
# How many simulation steps to record. Recording starts at step 0 and stops
# automatically once this many steps have been logged (steps only advance
# while unpaused, so pausing does not count against this budget).
# 0.016 between each step
RECORD_STEPS = _env_int("BOIDS_RECORD_STEPS", 37500)
FPS = 60  # used only for the "elapsed time" message printed when done

# If True, the whole program exits (closing the pygame window) the moment
# recording finishes. If False, the simulation keeps running/visible but
# no further rows are written to the log files.
QUIT_WHEN_RECORDING_DONE = _env_bool("BOIDS_QUIT_WHEN_RECORDING_DONE", False)

# ---------------------------------------------------------------------------
# Headless data-generation mode
# ---------------------------------------------------------------------------
# If True, no window is created and nothing is drawn -- the sim just runs
# the physics and writes to the log files as fast as it can (no 60fps cap),
# then exits automatically the moment RECORD_STEPS have been written.
# Use this when you only want boids_log.csv / boids_log.bin and don't care
# about watching the simulation.
HEADLESS_MODE = _env_bool("BOIDS_HEADLESS_MODE", False)