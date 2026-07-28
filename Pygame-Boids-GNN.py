import pygame  # loads Pygame Library
import sys     # loads pythons build in system tools
import random
import math
import csv
import struct

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
SEPARATION_RADIUS = 35
COHESION_RADIUS = 50
ALIGNMENT_RADIUS = 100

# Force weights -- how strongly each rule pulls on a boid's velocity.
SEPARATION_WEIGHT = 0.02
COHESION_WEIGHT = 0.001
ALIGNMENT_WEIGHT = 0.05

MIN_SPEED = 1
MAX_SPEED = 4

# ---------------------------------------------------------------------------
# Leader / Follower settings
# ---------------------------------------------------------------------------
NUM_LEADERS = 3                 # first N boids created are designated leaders
LEADER_FOLLOW_RADIUS = 250      # followers only react to a leader within this range
LEADER_FOLLOW_WEIGHT = 0.03     # how strongly a follower steers toward its nearest leader
LEADER_WANDER_STRENGTH = 0.15   # how sharply a leader's own wander heading can turn per step
LEADER_WANDER_FORCE = 0.12      # how strongly the wander heading pulls a leader's velocity
LEADER_COLOR = (255, 210, 60)
FOLLOWER_COLOR = (100, 200, 255)

# ---------------------------------------------------------------------------
# Predator settings
# ---------------------------------------------------------------------------
NUM_PREDATORS = 1
PREDATOR_AVOID_RADIUS = 120     # boids inside this radius of a predator will flee
PREDATOR_AVOID_WEIGHT = 3.0     # strength of the flee force
PREDATOR_CHASE_WEIGHT = 0.01    # how strongly a predator steers toward the nearest boid
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 3.5
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
BORDER_MODE = "wrap"            # "wrap" or "bounded"
BORDER_MARGIN = 100             # bounded mode: distance from edge where turning starts
BORDER_TURN_WEIGHT = 1.6        # bounded mode: how hard to steer away from the edge -- has
                                 # to be bigger than it looks like it should, since it needs
                                 # to be able to out-muscle a whole cluster of boids all
                                 # cohering toward each other right at the wall
BORDER_BOUNCE_KICK = 1.0        # bounded mode: minimum outward speed added on contact
                                 # with the wall itself -- prevents stalling in a corner
BORDER_EDGE_DRAG = 0.4          # bounded mode: how much along-wall (tangential) speed
                                 # bleeds off near a wall -- prevents gliding along the edge

WORLD_WIDTH = 900
WORLD_HEIGHT = 650

CSV_PATH = "boids_log.csv"
BIN_PATH = "boids_log.bin"

# How many simulation steps to record. Recording starts at step 0 and stops
# automatically once this many steps have been logged (steps only advance
# while unpaused, so pausing does not count against this budget).
# 0.016 between each step
RECORD_STEPS = 500
FPS = 60  # used only for the "elapsed time" message printed when done

# If True, the whole program exits (closing the pygame window) the moment
# recording finishes. If False, the simulation keeps running/visible but
# no further rows are written to the log files.
QUIT_WHEN_RECORDING_DONE = False

# ---------------------------------------------------------------------------
# Headless data-generation mode
# ---------------------------------------------------------------------------
# If True, no window is created and nothing is drawn -- the sim just runs
# the physics and writes to the log files as fast as it can (no 60fps cap),
# then exits automatically the moment RECORD_STEPS have been written.
# Use this when you only want boids_log.csv / boids_log.bin and don't care
# about watching the simulation.
HEADLESS_MODE = True

CSV_FIELDS = [
    "step", "entity_type", "entity_id", "x", "y", "x_vel", "y_vel",
    "pre_planned", "is_leader", "leader_id",
    "separation", "cohesion", "alignment", "predator_avoid",
    "border_mode",
]

# ---------------------------------------------------------------------------
# Binary format (little-endian), one record per entity (boid or predator)
# per step:
#
#   uint32   step
#   uint8    entity_type            (0 = boid, 1 = predator)
#   uint32   entity_id
#   float32  x
#   float32  y
#   float32  x_vel
#   float32  y_vel
#   uint8    pre_planned            (0 or 1)
#   uint8    is_leader              (0 or 1, always 0 for predators)
#   int32    leader_id              (-1 if not following a leader / not a boid)
#   uint8    border_mode            (0 = wrap, 1 = bounded)
#   uint16   separation_count
#   uint32[] separation_ids         (separation_count entries)
#   uint16   cohesion_count
#   uint32[] cohesion_ids
#   uint16   alignment_count
#   uint32[] alignment_ids
#   uint16   predator_avoid_count
#   uint32[] predator_avoid_ids
#
# There is no fixed record length (neighbor lists vary in size), so the
# file must be parsed sequentially from the start, reading each count
# before its id list.
# ---------------------------------------------------------------------------
RECORD_HEADER_FMT = "<IBIffffBBiB"  # step, entity_type, entity_id, x, y, x_vel,
                                    # y_vel, pre_planned, is_leader, leader_id,
                                    # border_mode

ENTITY_TYPE_BOID = 0
ENTITY_TYPE_PREDATOR = 1
BORDER_MODE_WRAP = 0
BORDER_MODE_BOUNDED = 1


def pack_record(step, entity_type, entity_id, x, y, x_vel, y_vel, pre_planned,
                 is_leader, leader_id, border_mode,
                 separation_ids, cohesion_ids, alignment_ids, predator_ids):
    entity_type_code = ENTITY_TYPE_BOID if entity_type == "boid" else ENTITY_TYPE_PREDATOR
    border_mode_code = BORDER_MODE_WRAP if border_mode == "wrap" else BORDER_MODE_BOUNDED

    data = struct.pack(
        RECORD_HEADER_FMT,
        step, entity_type_code, entity_id, x, y, x_vel, y_vel,
        1 if pre_planned else 0, 1 if is_leader else 0, leader_id,
        border_mode_code,
    )
    for ids in (separation_ids, cohesion_ids, alignment_ids, predator_ids):
        data += struct.pack("<H", len(ids))
        if ids:
            data += struct.pack(f"<{len(ids)}I", *ids)
    return data


def apply_world_border(entity, width=WORLD_WIDTH, height=WORLD_HEIGHT):
    """Either wrap the entity around the screen, or -- in bounded mode --
    steer it away from the edges and clamp it so it can never leave."""
    if BORDER_MODE == "wrap":
        if entity.x > width:
            entity.x = 0
        if entity.x < 0:
            entity.x = width
        if entity.y > height:
            entity.y = 0
        if entity.y < 0:
            entity.y = height
    else:  # "bounded" -- the edge of the world
        # soft steering: gently curve away from a wall as it's approached.
        # Alongside the push straight back into the field, also bleed off
        # whatever velocity runs *parallel* to that wall -- without this,
        # a boid arriving at a shallow angle just keeps its sideways speed
        # and glides along the edge instead of turning back in.
        if entity.x < BORDER_MARGIN:
            closeness = (BORDER_MARGIN - entity.x) / BORDER_MARGIN
            entity.vx += BORDER_TURN_WEIGHT * closeness
            entity.vy *= (1 - BORDER_EDGE_DRAG * closeness)
        elif entity.x > width - BORDER_MARGIN:
            closeness = (BORDER_MARGIN - (width - entity.x)) / BORDER_MARGIN
            entity.vx -= BORDER_TURN_WEIGHT * closeness
            entity.vy *= (1 - BORDER_EDGE_DRAG * closeness)

        if entity.y < BORDER_MARGIN:
            closeness = (BORDER_MARGIN - entity.y) / BORDER_MARGIN
            entity.vy += BORDER_TURN_WEIGHT * closeness
            entity.vx *= (1 - BORDER_EDGE_DRAG * closeness)
        elif entity.y > height - BORDER_MARGIN:
            closeness = (BORDER_MARGIN - (height - entity.y)) / BORDER_MARGIN
            entity.vy -= BORDER_TURN_WEIGHT * closeness
            entity.vx *= (1 - BORDER_EDGE_DRAG * closeness)

        # hard bounce right at the wall itself. This is what actually stops
        # entities from stalling/piling up in a corner: even if flocking or
        # leader-follow forces are pulling hard enough to cancel out the
        # soft steering above, this unconditionally reflects the velocity
        # away from the wall (with a small minimum kick so it can't settle
        # at exactly zero velocity against the edge).
        if entity.x <= 0:
            entity.x = 0
            entity.vx = abs(entity.vx) + BORDER_BOUNCE_KICK
        elif entity.x >= width:
            entity.x = width
            entity.vx = -abs(entity.vx) - BORDER_BOUNCE_KICK

        if entity.y <= 0:
            entity.y = 0
            entity.vy = abs(entity.vy) + BORDER_BOUNCE_KICK
        elif entity.y >= height:
            entity.y = height
            entity.vy = -abs(entity.vy) - BORDER_BOUNCE_KICK


# Setup
pygame.init()  # starts pygame always needed

if HEADLESS_MODE:
    # no window, no clock cap, no font -- just run the physics/logging loop
    screen = None
    clock = None
    hud_font = None
else:
    # creates a variable name screen and creates a window thats 900pixels wide and 650 in height
    screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont(None, 22)

    # names the window
    pygame.display.set_caption("Boids Model")


class Boid:

    def __init__(self, boid_id):
        self.id = boid_id
        self.x = random.uniform(0, WORLD_WIDTH)
        self.y = random.uniform(0, WORLD_HEIGHT)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.pre_planned = False  # not yet used, reserved for precomputed-path boids

        self.is_leader = False
        self.wander_angle = random.uniform(0, 2 * math.pi)
        self.leader_id = -1          # id of the leader this boid is currently following, -1 if none
        self.predator_avoid_ids = []  # predators currently within PREDATOR_AVOID_RADIUS

    def cohesion(self, boids):
        center_x = 0
        center_y = 0
        count = 0

        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < COHESION_RADIUS:
                center_x += other.x
                center_y += other.y
                count += 1

        if count > 0:
            center_x /= count
            center_y /= count
            self.vx += (center_x - self.x) * COHESION_WEIGHT
            self.vy += (center_y - self.y) * COHESION_WEIGHT

    def separation(self, boids):
        move_x = 0
        move_y = 0

        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < SEPARATION_RADIUS:
                move_x -= dx
                move_y -= dy

        self.vx += move_x * SEPARATION_WEIGHT
        self.vy += move_y * SEPARATION_WEIGHT

    def limit_speed(self, min_speed, max_speed):
        speed = (self.vx**2 + self.vy**2) ** 0.5    # total speed (distance formula)

        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        elif speed < min_speed:
            if speed == 0:
                # no direction to scale up from -- pick a random heading
                angle = random.uniform(0, 2 * math.pi)
                self.vx = math.cos(angle) * min_speed
                self.vy = math.sin(angle) * min_speed
            else:
                self.vx = (self.vx / speed) * min_speed
                self.vy = (self.vy / speed) * min_speed

    def alignment(self, boids):
        avg_vx = 0
        avg_vy = 0
        count = 0

        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < ALIGNMENT_RADIUS:
                avg_vx += other.vx
                avg_vy += other.vy
                count += 1

        if count > 0:
            avg_vx /= count
            avg_vy /= count
            self.vx += (avg_vx - self.vx) * ALIGNMENT_WEIGHT
            self.vy += (avg_vy - self.vy) * ALIGNMENT_WEIGHT

    def leader_wander(self):
        """Leaders aren't pulled by any other boid's leadership -- they just
        roam, gently changing heading over time, and the flock follows."""
        self.wander_angle += random.uniform(-LEADER_WANDER_STRENGTH, LEADER_WANDER_STRENGTH)
        self.vx += math.cos(self.wander_angle) * LEADER_WANDER_FORCE
        self.vy += math.sin(self.wander_angle) * LEADER_WANDER_FORCE

    def follow_leader(self, leaders):
        """Steer toward the nearest leader within LEADER_FOLLOW_RADIUS, if any."""
        nearest = None
        nearest_dist = None

        for leader in leaders:
            dx = leader.x - self.x
            dy = leader.y - self.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < LEADER_FOLLOW_RADIUS and (nearest_dist is None or distance < nearest_dist):
                nearest = leader
                nearest_dist = distance

        if nearest is not None:
            dx = nearest.x - self.x
            dy = nearest.y - self.y
            self.vx += dx * LEADER_FOLLOW_WEIGHT
            self.vy += dy * LEADER_FOLLOW_WEIGHT
            self.leader_id = nearest.id
        else:
            self.leader_id = -1

    def avoid_predators(self, predators):
        """Flee any predator within PREDATOR_AVOID_RADIUS. Returns the ids of
        predators currently being avoided, for logging."""
        ids = []
        for predator in predators:
            dx = self.x - predator.x
            dy = self.y - predator.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < PREDATOR_AVOID_RADIUS:
                ids.append(predator.id)
                if distance == 0:
                    distance = 0.1
                strength = (PREDATOR_AVOID_RADIUS - distance) / PREDATOR_AVOID_RADIUS
                self.vx += (dx / distance) * strength * PREDATOR_AVOID_WEIGHT
                self.vy += (dy / distance) * strength * PREDATOR_AVOID_WEIGHT
        return ids

    def avoid_obstacles(self, obstacles):
        for ox, oy, radius in obstacles:
            dx = self.x - ox
            dy = self.y - oy
            distance = (dx**2 + dy**2) ** 0.5
            danger_zone = radius + 30

            if distance < radius + 30:        # danger zone = radius + buffer
                if distance == 0:
                    distance = 0.1            # avoid dividing by zero
                strength = (danger_zone - distance) / danger_zone
                self.vx += (dx / distance) * strength * 2
                self.vy += (dy / distance) * strength * 2

    def enforce_no_overlap(self, obstacles):
        for ox, oy, radius in obstacles:
            dx = self.x - ox
            dy = self.y - oy
            distance = (dx**2 + dy**2) ** 0.5

            if distance < radius and distance > 0:
                # push the boid to sit exactly on the edge of the obstacle
                self.x = ox + (dx / distance) * radius
                self.y = oy + (dy / distance) * radius

    def update(self, boids, leaders, predators, obstacles):
        self.cohesion(boids)
        self.separation(boids)
        self.alignment(boids)

        if self.is_leader:
            self.leader_wander()
            self.leader_id = -1
        else:
            self.follow_leader(leaders)

        self.predator_avoid_ids = self.avoid_predators(predators)
        self.avoid_obstacles(obstacles)
        self.limit_speed(MIN_SPEED, MAX_SPEED)
        self.x += self.vx
        self.y += self.vy
        self.enforce_no_overlap(obstacles)
        apply_world_border(self)

    def draw(self, screen):
        x, y = int(self.x), int(self.y)
        angle = math.atan2(self.vy, self.vx)   # direction of travel, in radians

        size = 12 if self.is_leader else 10
        color = LEADER_COLOR if self.is_leader else FOLLOWER_COLOR

        tip   = (x + math.cos(angle) * size,           y + math.sin(angle) * size)
        left  = (x + math.cos(angle + 2.5) * size*0.6, y + math.sin(angle + 2.5) * size*0.6)
        right = (x + math.cos(angle - 2.5) * size*0.6, y + math.sin(angle - 2.5) * size*0.6)

        pygame.draw.polygon(screen, color, [tip, left, right])

    def neighbors_within(self, boids, radius):
        """Return a sorted list of boid_ids within `radius` of this boid."""
        ids = []
        for other in boids:
            if other is self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = (dx**2 + dy**2) ** 0.5
            if distance < radius:
                ids.append(other.id)
        ids.sort()
        return ids


class Predator:

    def __init__(self, predator_id):
        self.id = predator_id
        self.x = random.uniform(0, WORLD_WIDTH)
        self.y = random.uniform(0, WORLD_HEIGHT)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)

    def chase(self, boids):
        nearest = None
        nearest_dist = None

        for boid in boids:
            dx = boid.x - self.x
            dy = boid.y - self.y
            distance = (dx**2 + dy**2) ** 0.5
            if nearest_dist is None or distance < nearest_dist:
                nearest = boid
                nearest_dist = distance

        if nearest is not None:
            dx = nearest.x - self.x
            dy = nearest.y - self.y
            self.vx += dx * PREDATOR_CHASE_WEIGHT
            self.vy += dy * PREDATOR_CHASE_WEIGHT

    def avoid_obstacles(self, obstacles):
        for ox, oy, radius in obstacles:
            dx = self.x - ox
            dy = self.y - oy
            distance = (dx**2 + dy**2) ** 0.5
            danger_zone = radius + 30

            if distance < danger_zone:
                if distance == 0:
                    distance = 0.1
                strength = (danger_zone - distance) / danger_zone
                self.vx += (dx / distance) * strength * 2
                self.vy += (dy / distance) * strength * 2

    def limit_speed(self, min_speed, max_speed):
        speed = (self.vx**2 + self.vy**2) ** 0.5

        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        elif speed < min_speed:
            if speed == 0:
                angle = random.uniform(0, 2 * math.pi)
                self.vx = math.cos(angle) * min_speed
                self.vy = math.sin(angle) * min_speed
            else:
                self.vx = (self.vx / speed) * min_speed
                self.vy = (self.vy / speed) * min_speed

    def update(self, boids, obstacles):
        self.chase(boids)
        self.avoid_obstacles(obstacles)
        self.limit_speed(PREDATOR_MIN_SPEED, PREDATOR_MAX_SPEED)
        self.x += self.vx
        self.y += self.vy
        apply_world_border(self)

    def draw(self, screen):
        x, y = int(self.x), int(self.y)
        angle = math.atan2(self.vy, self.vx)

        size = 16
        tip   = (x + math.cos(angle) * size,             y + math.sin(angle) * size)
        left  = (x + math.cos(angle + 2.6) * size*0.7,   y + math.sin(angle + 2.6) * size*0.7)
        right = (x + math.cos(angle - 2.6) * size*0.7,   y + math.sin(angle - 2.6) * size*0.7)

        pygame.draw.polygon(screen, PREDATOR_COLOR, [tip, left, right])


def log_step(step, boids, predators, csv_writer, bin_file):
    """Write one row per boid and one row per predator (CSV), and one packed
    record per entity (binary)."""
    for boid in boids:
        separation_ids = boid.neighbors_within(boids, SEPARATION_RADIUS)
        cohesion_ids = boid.neighbors_within(boids, COHESION_RADIUS)
        alignment_ids = boid.neighbors_within(boids, ALIGNMENT_RADIUS)
        predator_ids = boid.predator_avoid_ids

        csv_writer.writerow({
            "step": step,
            "entity_type": "boid",
            "entity_id": boid.id,
            "x": f"{boid.x:.4f}",
            "y": f"{boid.y:.4f}",
            "x_vel": f"{boid.vx:.4f}",
            "y_vel": f"{boid.vy:.4f}",
            "pre_planned": int(boid.pre_planned),
            "is_leader": int(boid.is_leader),
            "leader_id": boid.leader_id,
            "separation": "-".join(str(i) for i in separation_ids),
            "cohesion": "-".join(str(i) for i in cohesion_ids),
            "alignment": "-".join(str(i) for i in alignment_ids),
            "predator_avoid": "-".join(str(i) for i in predator_ids),
            "border_mode": BORDER_MODE,
        })

        bin_file.write(pack_record(
            step, "boid", boid.id, boid.x, boid.y, boid.vx, boid.vy,
            boid.pre_planned, boid.is_leader, boid.leader_id, BORDER_MODE,
            separation_ids, cohesion_ids, alignment_ids, predator_ids,
        ))

    for predator in predators:
        csv_writer.writerow({
            "step": step,
            "entity_type": "predator",
            "entity_id": predator.id,
            "x": f"{predator.x:.4f}",
            "y": f"{predator.y:.4f}",
            "x_vel": f"{predator.vx:.4f}",
            "y_vel": f"{predator.vy:.4f}",
            "pre_planned": 0,
            "is_leader": 0,
            "leader_id": -1,
            "separation": "",
            "cohesion": "",
            "alignment": "",
            "predator_avoid": "",
            "border_mode": BORDER_MODE,
        })

        bin_file.write(pack_record(
            step, "predator", predator.id, predator.x, predator.y,
            predator.vx, predator.vy, False, False, -1, BORDER_MODE,
            [], [], [], [],
        ))


obstacles = []

# create 36 boids, first NUM_LEADERS of them are designated leaders
boids = [Boid(i) for i in range(36)]
for i in range(min(NUM_LEADERS, len(boids))):
    boids[i].is_leader = True

predators = [Predator(i) for i in range(NUM_PREDATORS)]

paused = False
step = 0
recording = True

csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
csv_writer.writeheader()
bin_file = open(BIN_PATH, "wb")


def stop_recording():
    """Close the log files. Safe to call more than once."""
    global recording
    if recording:
        csv_file.flush()
        bin_file.flush()
        csv_file.close()
        bin_file.close()
        recording = False
        print(f"Finished recording {step} steps ({step / FPS:.1f}s) to "
              f"{CSV_PATH} and {BIN_PATH}")


def shutdown():
    if recording:
        stop_recording()
    pygame.quit()
    sys.exit()


# Game loop
# keeps looping everything in the while loop
while True:

    if not HEADLESS_MODE:
        # collects a list of things that happen(mouse click, key press, closing the window)
        for event in pygame.event.get():

            # if the user clicks x close the window
            if event.type == pygame.QUIT:
                shutdown()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_x:
                    obstacles.clear()
                if event.key == pygame.K_b:
                    # toggle the world border mode live -- also affects the
                    # "border_mode" value written to the log files from now on
                    BORDER_MODE = "bounded" if BORDER_MODE == "wrap" else "wrap"
                if event.key == pygame.K_ESCAPE:
                    shutdown()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                if event.button == 1:                    # left click
                    obstacles.append((mouse_x, mouse_y, 40))

                elif event.button == 3:                  # right click
                    if obstacles:                         # only if there's at least one
                        closest = None
                        closest_dist = None

                        for obstacle in obstacles:
                            ox, oy, radius = obstacle
                            dist = ((ox - mouse_x)**2 + (oy - mouse_y)**2) ** 0.5

                            if closest_dist is None or dist < closest_dist:
                                closest = obstacle
                                closest_dist = dist

                        obstacles.remove(closest)

        # paints the entire window dark blue
        screen.fill((15, 20, 35))

        for ox, oy, radius in obstacles:
            pygame.draw.circle(screen, (200, 80, 80), (ox, oy), radius)

    if not paused:
        leaders = [b for b in boids if b.is_leader]

        for boid in boids:
            boid.update(boids, leaders, predators, obstacles)

        for predator in predators:
            predator.update(boids, obstacles)

        if recording:
            log_step(step, boids, predators, csv_writer, bin_file)
            step += 1

            if step >= RECORD_STEPS:
                stop_recording()
                # headless mode has no window to keep open, so once the
                # data we came here for is written, always exit
                if QUIT_WHEN_RECORDING_DONE or HEADLESS_MODE:
                    shutdown()
            elif step % FPS == 0:      # flush to disk roughly once a second
                csv_file.flush()
                bin_file.flush()

    if HEADLESS_MODE:
        # no rendering, no frame cap -- run as fast as possible
        continue

    for boid in boids:
        boid.draw(screen)

    for predator in predators:
        predator.draw(screen)

    hud_lines = [
        f"border: {BORDER_MODE}  (press B to toggle)",
        f"{'paused' if paused else 'running'}   step: {step}",
        f"leaders: {len(leaders) if not paused else sum(b.is_leader for b in boids)}   predators: {len(predators)}",
    ]
    for i, line in enumerate(hud_lines):
        surf = hud_font.render(line, True, (200, 200, 210))
        screen.blit(surf, (10, 10 + i * 18))

    # shows everything done on the screen
    pygame.display.flip()
    clock.tick(60)