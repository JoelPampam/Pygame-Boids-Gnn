import os


# ==========================================================
# Boids Behavior
# ==========================================================
SEPARATION_RADIUS = 35
COHESION_RADIUS = 50
ALIGNMENT_RADIUS = 100
SEPARATION_WEIGHT = 0.02
COHESION_WEIGHT = 0.001
ALIGNMENT_WEIGHT = 0.05
MIN_SPEED = 1
MAX_SPEED = 4
# ==========================================================
# Leader / Follower
# ==========================================================
NUM_LEADERS = 3
LEADER_FOLLOW_RADIUS = 250
LEADER_FOLLOW_WEIGHT = 0.03
LEADER_WANDER_STRENGTH = 0.15
LEADER_WANDER_FORCE = 0.12
LEADER_COLOR = (255,210,60)
FOLLOWER_COLOR = (100,200,255)
# ==========================================================
# Predator
# ==========================================================
NUM_PREDATORS = 3
PREDATOR_AVOID_RADIUS = 120
PREDATOR_AVOID_WEIGHT = 3.0
PREDATOR_CHASE_WEIGHT = 0.01
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 3.5
PREDATOR_COLOR = (220,40,40)
# ==========================================================
# World
# ==========================================================
WORLD_WIDTH = 900
WORLD_HEIGHT = 650
BORDER_MODE = "wrap"
BORDER_MARGIN = 100
BORDER_TURN_WEIGHT = 1.6
BORDER_BOUNCE_KICK = 1.0
BORDER_EDGE_DRAG = 0.4
# ==========================================================
# Simulation
# ==========================================================
RECORD_STEPS = 37500
FPS = 60
HEADLESS_MODE = False
QUIT_WHEN_RECORDING_DONE = False
# ==========================================================
# Files
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR,"boids_log.csv")
BIN_PATH = os.path.join(SCRIPT_DIR,"boids_log.bin")