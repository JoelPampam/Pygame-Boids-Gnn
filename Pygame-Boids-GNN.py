import pygame  # loads Pygame Library
import sys     # loads pythons build in system tools
import random
import math
import csv
import struct
import os
from config import *



CSV_FIELDS = [
    "step", "entity_type", "entity_id", "x", "y", "x_vel", "y_vel",
    "pre_planned", "is_leader", "leader_id", "waypoint_id",
    "separation", "cohesion", "alignment", "predator_avoid",
    "obstacle_avoid", "obstacle_contact",
    "radius", "event",
    "border_mode",
]
# "radius" and "event" are only populated for entity_type == "obstacle"
# rows (event is "created" or "removed"); they're blank everywhere else.
# Obstacles aren't static like waypoints and don't have a fixed count, so
# rather than re-logging every obstacle's position every step, each one
# gets exactly two rows -- a "created" row and (if it's ever removed) a
# "removed" row. The active obstacle set at any given step can be
# reconstructed from those intervals, filtered by entity_id.

# ---------------------------------------------------------------------------
# Binary format (little-endian), one record per entity (boid, predator, or
# obstacle) -- this mirrors the CSV field-for-field:
#
#   uint32   step
#   uint8    entity_type            (0 = boid, 1 = predator, 2 = obstacle)
#   uint32   entity_id
#   float32  x
#   float32  y
#   float32  x_vel
#   float32  y_vel
#   uint8    pre_planned            (0 or 1)
#   uint8    is_leader              (0 or 1, always 0 for predators/obstacles)
#   int32    leader_id              (-1 if not following a leader / not a boid)
#   int32    waypoint_id            (numbered goal spot currently being pursued:
#                                    a leader's own index into WAYPOINTS, or the
#                                    waypoint_id of the leader a follower is
#                                    currently following; -1 if not applicable /
#                                    not currently attached to a leader)
#   uint8    border_mode            (0 = wrap, 1 = bounded)
#   uint16   separation_count
#   uint32[] separation_ids         (separation_count entries)
#   uint16   cohesion_count
#   uint32[] cohesion_ids
#   uint16   alignment_count
#   uint32[] alignment_ids
#   uint16   predator_avoid_count
#   uint32[] predator_avoid_ids
#   uint16   obstacle_avoid_count     (obstacles currently within soft
#                                      avoidance range -- radius + 30)
#   uint32[] obstacle_avoid_ids
#   uint16   obstacle_contact_count   (obstacles currently in hard contact /
#                                      overlap -- boids only, always 0 for
#                                      predators, which have no collision
#                                      resolution against obstacles)
#   uint32[] obstacle_contact_ids
#   float32  radius                 (obstacle records only; 0 for boid/predator)
#   uint8    event                  (0 = n/a, 1 = created, 2 = removed --
#                                    obstacle records only; 0 for boid/predator)
#
# Obstacle records: emitted only twice per obstacle (on creation and, if it
# happens, on removal) rather than every step -- same as the CSV. Obstacle
# ids are stable and never reused, assigned in creation order. The active
# obstacle set at any step is reconstructed the same way as from the CSV:
# by pairing each obstacle_id's "created"/"removed" event records.
#
# There is no fixed record length (neighbor lists vary in size), so the
# file must be parsed sequentially from the start, reading each count
# before its id list.
# ---------------------------------------------------------------------------
RECORD_HEADER_FMT = "<IBIffffBBiiBfB"  # step, entity_type, entity_id, x, y, x_vel,
                                       # y_vel, pre_planned, is_leader, leader_id,
                                       # waypoint_id, border_mode, radius, event

ENTITY_TYPE_BOID = 0
ENTITY_TYPE_PREDATOR = 1
ENTITY_TYPE_OBSTACLE = 2
BORDER_MODE_WRAP = 0
BORDER_MODE_BOUNDED = 1
EVENT_NA = 0
EVENT_CREATED = 1
EVENT_REMOVED = 2


def pack_record(step, entity_type, entity_id, x, y, x_vel, y_vel, pre_planned,
                 is_leader, leader_id, waypoint_id, border_mode,
                 separation_ids, cohesion_ids, alignment_ids, predator_ids,
                 obstacle_avoid_ids, obstacle_contact_ids,
                 radius=0.0, event=None):
    entity_type_code = {
        "boid": ENTITY_TYPE_BOID,
        "predator": ENTITY_TYPE_PREDATOR,
        "obstacle": ENTITY_TYPE_OBSTACLE,
    }[entity_type]
    border_mode_code = BORDER_MODE_WRAP if border_mode == "wrap" else BORDER_MODE_BOUNDED
    event_code = {"created": EVENT_CREATED, "removed": EVENT_REMOVED}.get(event, EVENT_NA)

    data = struct.pack(
        RECORD_HEADER_FMT,
        step, entity_type_code, entity_id, x, y, x_vel, y_vel,
        1 if pre_planned else 0, 1 if is_leader else 0, leader_id,
        waypoint_id, border_mode_code, radius, event_code,
    )
    for ids in (separation_ids, cohesion_ids, alignment_ids, predator_ids,
                obstacle_avoid_ids, obstacle_contact_ids):
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
        self.obstacle_avoid_ids = []    # obstacles currently within soft avoidance range
        self.obstacle_contact_ids = []  # obstacles currently in hard contact/overlap

        self.waypoint_index = 0       # only advanced/used if this boid is a leader
        self.target_waypoint_id = -1  # what actually gets logged -- see update()

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

    def seek_waypoint(self, waypoints):
        """Steer toward this leader's current target waypoint. Once within
        WAYPOINT_RADIUS of it, advance to the next one in the list (looping
        back to the start if WAYPOINT_LOOP is on, otherwise holding at the
        last waypoint). Only meaningful for leaders -- followers get their
        target_waypoint_id from whichever leader they're following instead."""
        if not waypoints:
            return

        target_x, target_y = waypoints[self.waypoint_index]
        dx = target_x - self.x
        dy = target_y - self.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance < WAYPOINT_RADIUS:
            if self.waypoint_index < len(waypoints) - 1:
                self.waypoint_index += 1
            elif WAYPOINT_LOOP:
                self.waypoint_index = 0
            # else: last waypoint and not looping -- just sit near it
        else:
            self.vx += dx * WAYPOINT_WEIGHT
            self.vy += dy * WAYPOINT_WEIGHT

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
            self.target_waypoint_id = nearest.waypoint_index
        else:
            self.leader_id = -1
            self.target_waypoint_id = -1

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
        """Soft steering away from nearby obstacles. Returns the ids of
        obstacles currently within the danger zone, for logging."""
        ids = []
        for obstacle in obstacles:
            dx = self.x - obstacle.x
            dy = self.y - obstacle.y
            distance = (dx**2 + dy**2) ** 0.5
            danger_zone = obstacle.radius + 30

            if distance < danger_zone:        # danger zone = radius + buffer
                ids.append(obstacle.id)
                if distance == 0:
                    distance = 0.1            # avoid dividing by zero
                strength = (danger_zone - distance) / danger_zone
                self.vx += (dx / distance) * strength * 2
                self.vy += (dy / distance) * strength * 2
        return ids

    def enforce_no_overlap(self, obstacles):
        """Hard collision resolution -- clamp the boid to the obstacle's
        edge if it's overlapping. Returns the ids of obstacles currently in
        contact, for logging. This is a distinct interaction from the soft
        steering in avoid_obstacles: a boid can be in the danger zone
        without being in contact, but not vice versa."""
        ids = []
        for obstacle in obstacles:
            dx = self.x - obstacle.x
            dy = self.y - obstacle.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < obstacle.radius and distance > 0:
                # push the boid to sit exactly on the edge of the obstacle
                self.x = obstacle.x + (dx / distance) * obstacle.radius
                self.y = obstacle.y + (dy / distance) * obstacle.radius
                ids.append(obstacle.id)
        return ids

    def update(self, boids, leaders, predators, obstacles):
        self.cohesion(boids)
        self.separation(boids)
        self.alignment(boids)

        if self.is_leader:
            self.leader_wander()
            self.seek_waypoint(WAYPOINTS)
            self.leader_id = -1
            self.target_waypoint_id = self.waypoint_index
        else:
            self.follow_leader(leaders)

        self.predator_avoid_ids = self.avoid_predators(predators)
        self.obstacle_avoid_ids = self.avoid_obstacles(obstacles)
        self.limit_speed(MIN_SPEED, MAX_SPEED)
        self.x += self.vx
        self.y += self.vy
        self.obstacle_contact_ids = self.enforce_no_overlap(obstacles)
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
        self.obstacle_avoid_ids = []  # obstacles currently within soft avoidance range

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
        """Soft steering away from nearby obstacles. Returns the ids of
        obstacles currently within the danger zone, for logging."""
        ids = []
        for obstacle in obstacles:
            dx = self.x - obstacle.x
            dy = self.y - obstacle.y
            distance = (dx**2 + dy**2) ** 0.5
            danger_zone = obstacle.radius + 30

            if distance < danger_zone:
                ids.append(obstacle.id)
                if distance == 0:
                    distance = 0.1
                strength = (danger_zone - distance) / danger_zone
                self.vx += (dx / distance) * strength * 2
                self.vy += (dy / distance) * strength * 2
        return ids

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
        self.obstacle_avoid_ids = self.avoid_obstacles(obstacles)
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


class Obstacle:
    """A user-placed obstacle with a stable id. Ids are assigned in creation
    order and never reused, so they stay valid as foreign keys into
    obstacles_log.csv even after the obstacle is removed from the live
    simulation."""

    def __init__(self, obstacle_id, x, y, radius):
        self.id = obstacle_id
        self.x = x
        self.y = y
        self.radius = radius


def draw_waypoints(screen, waypoints, font):
    """Draw each waypoint as a numbered ring/dot so the route is visible."""
    for i, (wx, wy) in enumerate(waypoints):
        pygame.draw.circle(screen, WAYPOINT_RING_COLOR, (int(wx), int(wy)), WAYPOINT_RADIUS, 2)
        pygame.draw.circle(screen, WAYPOINT_COLOR, (int(wx), int(wy)), 5)
        label = font.render(str(i), True, (230, 230, 230))
        screen.blit(label, (wx + 8, wy - 8))


def log_step(step, boids, predators, csv_writer, bin_file):
    """Write one row per boid and one row per predator (CSV), and one packed
    record per entity (binary)."""
    for boid in boids:
        separation_ids = boid.neighbors_within(boids, SEPARATION_RADIUS)
        cohesion_ids = boid.neighbors_within(boids, COHESION_RADIUS)
        alignment_ids = boid.neighbors_within(boids, ALIGNMENT_RADIUS)
        predator_ids = boid.predator_avoid_ids
        obstacle_avoid_ids = boid.obstacle_avoid_ids
        obstacle_contact_ids = boid.obstacle_contact_ids

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
            "waypoint_id": boid.target_waypoint_id,
            "separation": "-".join(str(i) for i in separation_ids),
            "cohesion": "-".join(str(i) for i in cohesion_ids),
            "alignment": "-".join(str(i) for i in alignment_ids),
            "predator_avoid": "-".join(str(i) for i in predator_ids),
            "obstacle_avoid": "-".join(str(i) for i in obstacle_avoid_ids),
            "obstacle_contact": "-".join(str(i) for i in obstacle_contact_ids),
            "border_mode": BORDER_MODE,
        })

        bin_file.write(pack_record(
            step, "boid", boid.id, boid.x, boid.y, boid.vx, boid.vy,
            boid.pre_planned, boid.is_leader, boid.leader_id,
            boid.target_waypoint_id, BORDER_MODE,
            separation_ids, cohesion_ids, alignment_ids, predator_ids,
            obstacle_avoid_ids, obstacle_contact_ids,
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
            "waypoint_id": -1,
            "separation": "",
            "cohesion": "",
            "alignment": "",
            "predator_avoid": "",
            "obstacle_avoid": "-".join(str(i) for i in predator.obstacle_avoid_ids),
            "obstacle_contact": "",
            "border_mode": BORDER_MODE,
        })

        bin_file.write(pack_record(
            step, "predator", predator.id, predator.x, predator.y,
            predator.vx, predator.vy, False, False, -1, -1, BORDER_MODE,
            [], [], [], [],
            predator.obstacle_avoid_ids, [],
        ))


obstacles = []          # active Obstacle instances only
next_obstacle_id = 0    # incrementing counter -- ids are never reused


def spawn_obstacle(x, y, radius, step):
    """Create a new obstacle, give it a stable id, and log its creation as
    a row in the main CSV and a record in the main bin file
    (entity_type == 'obstacle')."""
    global next_obstacle_id
    obstacle = Obstacle(next_obstacle_id, x, y, radius)
    next_obstacle_id += 1
    obstacles.append(obstacle)
    if recording:
        csv_writer.writerow({
            "step": step, "entity_type": "obstacle", "entity_id": obstacle.id,
            "x": f"{x:.4f}", "y": f"{y:.4f}", "radius": radius, "event": "created",
        })
        bin_file.write(pack_record(
            step, "obstacle", obstacle.id, x, y, 0.0, 0.0,
            False, False, -1, -1, BORDER_MODE,
            [], [], [], [], [], [],
            radius=radius, event="created",
        ))
    return obstacle


def remove_obstacle(obstacle, step):
    """Remove an obstacle from the live simulation and log its removal as a
    row in the main CSV and a record in the main bin file."""
    obstacles.remove(obstacle)
    if recording:
        csv_writer.writerow({
            "step": step, "entity_type": "obstacle", "entity_id": obstacle.id,
            "x": f"{obstacle.x:.4f}", "y": f"{obstacle.y:.4f}",
            "radius": obstacle.radius, "event": "removed",
        })
        bin_file.write(pack_record(
            step, "obstacle", obstacle.id, obstacle.x, obstacle.y, 0.0, 0.0,
            False, False, -1, -1, BORDER_MODE,
            [], [], [], [], [], [],
            radius=obstacle.radius, event="removed",
        ))


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
                    for obstacle in list(obstacles):
                        remove_obstacle(obstacle, step)
                if event.key == pygame.K_b:
                    # toggle the world border mode live -- also affects the
                    # "border_mode" value written to the log files from now on
                    BORDER_MODE = "bounded" if BORDER_MODE == "wrap" else "wrap"
                if event.key == pygame.K_ESCAPE:
                    shutdown()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                if event.button == 1:                    # left click
                    spawn_obstacle(mouse_x, mouse_y, OBSTACLE_RADIUS, step)

                elif event.button == 3:                  # right click
                    if obstacles:                         # only if there's at least one
                        closest = None
                        closest_dist = None

                        for obstacle in obstacles:
                            dist = ((obstacle.x - mouse_x)**2 + (obstacle.y - mouse_y)**2) ** 0.5

                            if closest_dist is None or dist < closest_dist:
                                closest = obstacle
                                closest_dist = dist

                        remove_obstacle(closest, step)

        # paints the entire window dark blue
        screen.fill((15, 20, 35))

        draw_waypoints(screen, WAYPOINTS, hud_font)

        for obstacle in obstacles:
            pygame.draw.circle(screen, (200, 80, 80), (obstacle.x, obstacle.y), obstacle.radius)

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