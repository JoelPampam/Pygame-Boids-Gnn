import os
import torch
from torch.utils.data import Dataset


# ============================================================
# WORLD SIZE
# ============================================================
# Based on the simulation coordinates we have seen.
# We will later move these into config/data automatically.
# ============================================================

WORLD_WIDTH = 900.0
WORLD_HEIGHT = 650.0


# ============================================================
# WRAP-AROUND RELATIVE POSITION
# ============================================================

def wrapped_difference(delta, world_size):
    """
    Find the shortest displacement in a wrap-around world.

    Example:
        x_i = 899
        x_j = 1

        Normal difference:
            1 - 899 = -898

        Wrapped difference:
            +2
    """

    half = world_size / 2.0

    delta = torch.where(
        delta > half,
        delta - world_size,
        delta
    )

    delta = torch.where(
        delta < -half,
        delta + world_size,
        delta
    )

    return delta


# ============================================================
# BUILD ONE GRAPH
# ============================================================

def build_graph(run, timestep):
    """
    Convert one timestep into a fully connected directed graph.

    Vectorized implementation:
        no Python loops over individual edges.

    Edge convention:
        sender j -> receiver i
    """

    # ========================================================
    # GET STATE
    # ========================================================

    positions = run["positions"][timestep].float()
    velocities = run["velocities"][timestep].float()

    entity_types = run["entity_types"]
    is_leader = run["is_leader"][timestep].float()

    target = run["delta_velocity"][timestep].float()

    num_nodes = positions.shape[0]


    # ========================================================
    # NODE FEATURES
    # ========================================================

    # Normalize position
    normalized_positions = positions.clone()

    normalized_positions[:, 0] /= WORLD_WIDTH
    normalized_positions[:, 1] /= WORLD_HEIGHT

    # Normalize velocity
    normalized_velocities = velocities / 5.0

    # Entity type
    is_boid = (entity_types == 0).float()
    is_predator = (entity_types == 1).float()

    # Construct:
    #
    # [x, y, vx, vy, is_boid, is_predator, is_leader]
    #
    node_features = torch.cat(
        [
            normalized_positions,
            normalized_velocities,
            is_boid.unsqueeze(1),
            is_predator.unsqueeze(1),
            is_leader.unsqueeze(1)
        ],
        dim=1
    )


    # ========================================================
    # FULLY CONNECTED DIRECTED EDGE INDEX
    # ========================================================

    nodes = torch.arange(num_nodes)

    # Cartesian product:
    #
    # receiver, sender
    #
    receiver_grid, sender_grid = torch.meshgrid(
        nodes,
        nodes,
        indexing="ij"
    )

    receivers = receiver_grid.reshape(-1)
    senders = sender_grid.reshape(-1)

    # Remove:
    #
    # i -> i
    #
    mask = senders != receivers

    senders = senders[mask]
    receivers = receivers[mask]

    # Shape:
    #
    # [2, 1332]
    #
    edge_index = torch.stack(
        [
            senders,
            receivers
        ],
        dim=0
    )


    # ========================================================
    # RELATIVE POSITION
    # ========================================================

    relative_position = (
        positions[senders]
        - positions[receivers]
    )

    dx = relative_position[:, 0]
    dy = relative_position[:, 1]


    # ========================================================
    # WRAP-AROUND CORRECTION
    # ========================================================

    dx = wrapped_difference(
        dx,
        WORLD_WIDTH
    )

    dy = wrapped_difference(
        dy,
        WORLD_HEIGHT
    )


    # ========================================================
    # RELATIVE VELOCITY
    # ========================================================

    relative_velocity = (
        velocities[senders]
        - velocities[receivers]
    )

    dvx = relative_velocity[:, 0]
    dvy = relative_velocity[:, 1]


    # ========================================================
    # DISTANCE
    # ========================================================

    distance = torch.sqrt(
        dx ** 2 +
        dy ** 2
    )


    # ========================================================
    # NORMALIZE EDGE FEATURES
    # ========================================================

    edge_features = torch.stack(
        [
            dx / WORLD_WIDTH,
            dy / WORLD_HEIGHT,
            dvx / 5.0,
            dvy / 5.0,
            distance / max(
                WORLD_WIDTH,
                WORLD_HEIGHT
            )
        ],
        dim=1
    )


    # ========================================================
    # BOID MASK
    # ========================================================

    boid_mask = (
        entity_types == 0
    )


    # ========================================================
    # RETURN GRAPH
    # ========================================================

    graph = {
        "x": node_features,

        "edge_index": edge_index,
        "edge_attr": edge_features,

        "y": target,

        "boid_mask": boid_mask,

        "entity_types": entity_types,
        "entity_ids": run["entity_ids"],

        "timestep": timestep
    }

    return graph

# ============================================================
# DATASET
# ============================================================

class BoidsGraphDataset(Dataset):

    def __init__(self, pt_path):

        print(f"Loading {pt_path}")

        self.run = torch.load(
            pt_path,
            weights_only=False
        )

        # Last frame has no delta_velocity target.
        self.length = (
            self.run["delta_velocity"].shape[0]
        )

        print(
            f"Loaded {self.length} graph samples."
        )

    def __len__(self):

        return self.length

    def __getitem__(self, index):

        return build_graph(
            self.run,
            index
        )


# ============================================================
# TEST
# ============================================================

def main():

    path = os.path.join(
        "Data",
        "dataset_1",
        "processed",
        "boids_log_1.pt"
    )

    dataset = BoidsGraphDataset(path)

    graph = dataset[0]

    print()
    print("=" * 60)
    print("GRAPH TEST")
    print("=" * 60)

    print()
    print("Number of graph samples:")
    print(len(dataset))

    print()
    print("Node features shape:")
    print(graph["x"].shape)

    print()
    print("Edge index shape:")
    print(graph["edge_index"].shape)

    print()
    print("Edge features shape:")
    print(graph["edge_attr"].shape)

    print()
    print("Target shape:")
    print(graph["y"].shape)

    print()
    print("Boid mask shape:")
    print(graph["boid_mask"].shape)

    print()
    print("Number of boids:")
    print(
        graph["boid_mask"].sum().item()
    )

    print()
    print("First node:")
    print(graph["x"][0])

    print()
    print("First 5 edges:")
    print(
        graph["edge_index"][:, :5]
    )

    print()
    print("First 5 edge features:")
    print(
        graph["edge_attr"][:5]
    )

    print()
    print("First target delta-v:")
    print(graph["y"][0])

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()