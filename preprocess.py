import csv
import json
import os
import sys

import torch


# ============================================================
# ENTITY TYPES
# ============================================================

ENTITY_TYPE = {
    "boid": 0,
    "predator": 1,
    "obstacle": 2,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_id_list(value):
    """
    Convert:
        "3-6-10-12"
    into:
        [3, 6, 10, 12]

    Blank fields become [].
    """

    if value is None:
        return []

    value = value.strip()

    if value == "":
        return []

    return [int(x) for x in value.split("-")]


def load_config(dataset_dir):
    config_path = os.path.join(dataset_dir, "config.json")

    with open(config_path, "r") as f:
        manifest = json.load(f)

    return manifest


# ============================================================
# LOAD RAW CSV
# ============================================================

def load_csv(csv_path):

    frames = {}
    obstacle_events = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            step = int(row["step"])
            entity_type = row["entity_type"].strip()

            # ------------------------------------------------
            # Obstacles are event-based, so save them separately
            # ------------------------------------------------

            if entity_type == "obstacle":

                obstacle_events.append({
                    "step": step,
                    "entity_id": int(row["entity_id"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "radius": float(row["radius"]),
                    "event": row["event"].strip(),
                })

                continue

            # ------------------------------------------------
            # Boids / predators
            # ------------------------------------------------

            entity = {
                "entity_type": entity_type,
                "entity_id": int(row["entity_id"]),

                "x": float(row["x"]),
                "y": float(row["y"]),

                "vx": float(row["x_vel"]),
                "vy": float(row["y_vel"]),

                "is_leader": int(row["is_leader"]),

                "leader_id": int(row["leader_id"]),
                "waypoint_id": int(row["waypoint_id"]),

                # Ground-truth behavior labels.
                # These will NOT be given to the GNN.
                "separation": parse_id_list(row["separation"]),
                "cohesion": parse_id_list(row["cohesion"]),
                "alignment": parse_id_list(row["alignment"]),
                "predator_avoid": parse_id_list(row["predator_avoid"]),
                "obstacle_avoid": parse_id_list(row["obstacle_avoid"]),
                "obstacle_contact": parse_id_list(row["obstacle_contact"]),

                "border_mode": row["border_mode"].strip(),
            }

            if step not in frames:
                frames[step] = []

            frames[step].append(entity)

    return frames, obstacle_events


# ============================================================
# CONVERT ONE RUN INTO TENSORS
# ============================================================

def process_run(csv_path, dataset_dir):

    print(f"\nProcessing:")
    print(f"  {csv_path}")

    manifest = load_config(dataset_dir)
    config = manifest["config"]

    frames, obstacle_events = load_csv(csv_path)

    steps = sorted(frames.keys())

    if len(steps) < 2:
        raise RuntimeError("Run does not contain enough timesteps.")

    # --------------------------------------------------------
    # Determine entity ordering from first frame
    # --------------------------------------------------------

    first_frame = frames[steps[0]]

    first_frame = sorted(
        first_frame,
        key=lambda e: (
            ENTITY_TYPE[e["entity_type"]],
            e["entity_id"]
        )
    )

    entity_keys = [
        (e["entity_type"], e["entity_id"])
        for e in first_frame
    ]

    num_entities = len(entity_keys)
    num_steps = len(steps)

    print(f"  Timesteps: {num_steps}")
    print(f"  Dynamic entities: {num_entities}")

    # --------------------------------------------------------
    # Allocate tensors
    # --------------------------------------------------------

    positions = torch.zeros(
        (num_steps, num_entities, 2),
        dtype=torch.float32
    )

    velocities = torch.zeros(
        (num_steps, num_entities, 2),
        dtype=torch.float32
    )

    is_leader = torch.zeros(
        (num_steps, num_entities),
        dtype=torch.float32
    )

    entity_types = torch.zeros(
        num_entities,
        dtype=torch.long
    )

    entity_ids = torch.zeros(
        num_entities,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Ground-truth labels
    #
    # Keep these as Python structures for now because each
    # entity can have a different number of neighbors.
    # --------------------------------------------------------

    ground_truth = []

    # --------------------------------------------------------
    # Process every timestep
    # --------------------------------------------------------

    for t_index, step in enumerate(steps):

        frame = frames[step]

        lookup = {
            (e["entity_type"], e["entity_id"]): e
            for e in frame
        }

        frame_ground_truth = []

        for entity_index, key in enumerate(entity_keys):

            if key not in lookup:
                raise RuntimeError(
                    f"Entity {key} missing at step {step}"
                )

            entity = lookup[key]

            positions[t_index, entity_index, 0] = entity["x"]
            positions[t_index, entity_index, 1] = entity["y"]

            velocities[t_index, entity_index, 0] = entity["vx"]
            velocities[t_index, entity_index, 1] = entity["vy"]

            is_leader[t_index, entity_index] = entity["is_leader"]

            if t_index == 0:
                entity_types[entity_index] = ENTITY_TYPE[
                    entity["entity_type"]
                ]

                entity_ids[entity_index] = entity["entity_id"]

            frame_ground_truth.append({
                "entity_type": entity["entity_type"],
                "entity_id": entity["entity_id"],

                "leader_id": entity["leader_id"],
                "waypoint_id": entity["waypoint_id"],

                "separation": entity["separation"],
                "cohesion": entity["cohesion"],
                "alignment": entity["alignment"],
                "predator_avoid": entity["predator_avoid"],
                "obstacle_avoid": entity["obstacle_avoid"],
                "obstacle_contact": entity["obstacle_contact"],
            })

        ground_truth.append(frame_ground_truth)

    # --------------------------------------------------------
    # Calculate training target:
    #
    #       delta_v(t) = v(t+1) - v(t)
    #
    # Shape:
    #       [T-1, N, 2]
    # --------------------------------------------------------

    delta_velocity = (
        velocities[1:] -
        velocities[:-1]
    )

    # --------------------------------------------------------
    # Final processed object
    # --------------------------------------------------------

    processed = {
        "positions": positions,
        "velocities": velocities,
        "delta_velocity": delta_velocity,

        "entity_types": entity_types,
        "entity_ids": entity_ids,
        "is_leader": is_leader,

        "steps": torch.tensor(
            steps,
            dtype=torch.long
        ),

        "obstacle_events": obstacle_events,

        # DO NOT feed these to the model.
        "ground_truth": ground_truth,

        # Hidden simulator parameters.
        # DO NOT feed these to the first model.
        "config": config,
    }

    return processed


# ============================================================
# PROCESS ENTIRE DATASET DIRECTORY
# ============================================================

def process_dataset(dataset_dir):

    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.isdir(dataset_dir):
        raise RuntimeError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    config_path = os.path.join(
        dataset_dir,
        "config.json"
    )

    if not os.path.isfile(config_path):
        raise RuntimeError(
            f"No config.json found in {dataset_dir}"
        )

    processed_dir = os.path.join(
        dataset_dir,
        "processed"
    )

    os.makedirs(
        processed_dir,
        exist_ok=True
    )

    csv_files = [
        name
        for name in os.listdir(dataset_dir)
        if name.startswith("boids_log_")
        and name.endswith(".csv")
    ]

    csv_files.sort()

    if not csv_files:
        raise RuntimeError(
            "No boids_log_*.csv files found."
        )

    print("=" * 60)
    print("BOIDS -> GNN PREPROCESSOR")
    print("=" * 60)

    for csv_name in csv_files:

        csv_path = os.path.join(
            dataset_dir,
            csv_name
        )

        run_name = os.path.splitext(
            csv_name
        )[0]

        output_path = os.path.join(
            processed_dir,
            run_name + ".pt"
        )

        processed = process_run(
            csv_path,
            dataset_dir
        )

        torch.save(
            processed,
            output_path
        )

        print(f"  Saved:")
        print(f"  {output_path}")

        print(
            "  positions:",
            tuple(processed["positions"].shape)
        )

        print(
            "  velocities:",
            tuple(processed["velocities"].shape)
        )

        print(
            "  delta_velocity:",
            tuple(processed["delta_velocity"].shape)
        )

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python preprocess.py Data/dataset_1"
        )

        sys.exit(1)

    dataset_dir = sys.argv[1]

    process_dataset(dataset_dir)


if __name__ == "__main__":
    main()