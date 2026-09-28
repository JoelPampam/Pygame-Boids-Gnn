import torch

from dataset import BoidsGraphDataset
from model import BoidsGraphNet


MODEL_PATH = "boids_gnn_v2.pt"
DATA_PATH = "Data/dataset_1/processed/boids_log_2.pt"

EPS = 1e-8


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    """
    Returns cosine similarity between two 2D vectors.

        +1 = same direction
         0 = perpendicular / unrelated
        -1 = opposite direction
    """

    denominator = (
        torch.linalg.vector_norm(a)
        * torch.linalg.vector_norm(b)
    )

    if denominator < EPS:
        return None

    return float(
        torch.dot(a, b) / denominator
    )


# ============================================================
# GROUND-TRUTH RULE LOOKUP
# ============================================================

def get_rule_ids(
    ground_truth,
    timestep,
    entity_id,
    rule_name
):
    """
    Get simulator neighbor IDs for one boid.

    These labels are used ONLY for evaluation.
    """

    frame = ground_truth[timestep]

    for entity in frame:

        if (
            entity["entity_type"] == "boid"
            and entity["entity_id"] == entity_id
        ):
            return entity.get(
                rule_name,
                []
            )

    return []


# ============================================================
# SUMMARY
# ============================================================

def summarize(name, values):

    print()
    print(name)
    print("-" * 60)

    print(
        f"Samples: {len(values)}"
    )

    if len(values) == 0:
        print("No valid samples.")
        return

    values = torch.tensor(
        values,
        dtype=torch.float32
    )

    mean = values.mean().item()
    median = values.median().item()

    positive = (
        (values > 0).float().mean().item()
        * 100
    )

    strong_positive = (
        (values > 0.5).float().mean().item()
        * 100
    )

    print(
        f"Mean cosine similarity:   "
        f"{mean:.6f}"
    )

    print(
        f"Median cosine similarity: "
        f"{median:.6f}"
    )

    print(
        f"Positive direction:       "
        f"{positive:.2f}%"
    )

    print(
        f"Cosine > 0.5:             "
        f"{strong_positive:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LEARNED INTERACTION DIRECTION ANALYSIS")
    print("=" * 60)

    dataset = BoidsGraphDataset(
        DATA_PATH
    )

    model = BoidsGraphNet()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            weights_only=True
        )
    )

    model.eval()

    ground_truth = dataset.run[
        "ground_truth"
    ]

    results = {
        "separation": [],
        "cohesion": [],
        "alignment": [],
        "predator_avoid": []
    }

    print()
    print(
        "Analyzing learned interaction directions..."
    )

    # ========================================================
    # ANALYZE
    # ========================================================

    with torch.no_grad():

        for timestep in range(len(dataset)):

            if timestep % 500 == 0:

                print(
                    f"Timestep "
                    f"{timestep}/{len(dataset)}"
                )

            graph = dataset[timestep]

            _, interactions = model(
                graph["x"],
                graph["edge_index"],
                graph["edge_attr"]
            )

            senders = graph[
                "edge_index"
            ][0]

            receivers = graph[
                "edge_index"
            ][1]

            entity_types = graph[
                "entity_types"
            ]

            entity_ids = graph[
                "entity_ids"
            ]

            # Original unnormalized state
            positions = dataset.run[
                "positions"
            ][timestep]

            velocities = dataset.run[
                "velocities"
            ][timestep]

            # =================================================
            # LOOP THROUGH EDGES
            # =================================================

            for edge_idx in range(
                interactions.shape[0]
            ):

                sender_index = int(
                    senders[edge_idx]
                )

                receiver_index = int(
                    receivers[edge_idx]
                )

                # Only analyze effects ON boids
                if int(
                    entity_types[
                        receiver_index
                    ]
                ) != 0:
                    continue

                sender_type = int(
                    entity_types[
                        sender_index
                    ]
                )

                sender_id = int(
                    entity_ids[
                        sender_index
                    ]
                )

                receiver_id = int(
                    entity_ids[
                        receiver_index
                    ]
                )

                message = interactions[
                    edge_idx
                ]

                # =============================================
                # WRAPPED RELATIVE POSITION
                #
                # sender - receiver
                # =============================================

                dx = (
                    positions[
                        sender_index, 0
                    ]
                    -
                    positions[
                        receiver_index, 0
                    ]
                )

                dy = (
                    positions[
                        sender_index, 1
                    ]
                    -
                    positions[
                        receiver_index, 1
                    ]
                )

                # Wrap correction
                if dx > 450:
                    dx -= 900
                elif dx < -450:
                    dx += 900

                if dy > 325:
                    dy -= 650
                elif dy < -325:
                    dy += 650

                relative_position = torch.tensor(
                    [dx, dy],
                    dtype=torch.float32
                )

                # =============================================
                # RELATIVE VELOCITY
                #
                # sender velocity - receiver velocity
                # =============================================

                relative_velocity = (
                    velocities[sender_index]
                    -
                    velocities[receiver_index]
                )

                # =============================================
                # BOID -> BOID
                # =============================================

                if sender_type == 0:

                    # -----------------------------------------
                    # SEPARATION
                    #
                    # Expected direction:
                    # AWAY from sender.
                    #
                    # sender-relative vector = r
                    # away = -r
                    # -----------------------------------------

                    separation_ids = get_rule_ids(
                        ground_truth,
                        timestep,
                        receiver_id,
                        "separation"
                    )

                    if sender_id in separation_ids:

                        expected = (
                            -relative_position
                        )

                        similarity = (
                            cosine_similarity(
                                message,
                                expected
                            )
                        )

                        if similarity is not None:
                            results[
                                "separation"
                            ].append(
                                similarity
                            )

                    # -----------------------------------------
                    # COHESION
                    #
                    # Expected direction:
                    # TOWARD sender.
                    # -----------------------------------------

                    cohesion_ids = get_rule_ids(
                        ground_truth,
                        timestep,
                        receiver_id,
                        "cohesion"
                    )

                    if sender_id in cohesion_ids:

                        expected = (
                            relative_position
                        )

                        similarity = (
                            cosine_similarity(
                                message,
                                expected
                            )
                        )

                        if similarity is not None:
                            results[
                                "cohesion"
                            ].append(
                                similarity
                            )

                    # -----------------------------------------
                    # ALIGNMENT
                    #
                    # Expected direction:
                    #
                    # v_sender - v_receiver
                    # -----------------------------------------

                    alignment_ids = get_rule_ids(
                        ground_truth,
                        timestep,
                        receiver_id,
                        "alignment"
                    )

                    if sender_id in alignment_ids:

                        expected = (
                            relative_velocity
                        )

                        similarity = (
                            cosine_similarity(
                                message,
                                expected
                            )
                        )

                        if similarity is not None:
                            results[
                                "alignment"
                            ].append(
                                similarity
                            )

                # =============================================
                # PREDATOR -> BOID
                # =============================================

                elif sender_type == 1:

                    predator_ids = get_rule_ids(
                        ground_truth,
                        timestep,
                        receiver_id,
                        "predator_avoid"
                    )

                    if sender_id in predator_ids:

                        # Boid should move AWAY
                        # from predator.
                        expected = (
                            -relative_position
                        )

                        similarity = (
                            cosine_similarity(
                                message,
                                expected
                            )
                        )

                        if similarity is not None:
                            results[
                                "predator_avoid"
                            ].append(
                                similarity
                            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("DIRECTION RESULTS")
    print("=" * 60)

    summarize(
        "SEPARATION — expected away from neighbor",
        results["separation"]
    )

    summarize(
        "COHESION — expected toward neighbor",
        results["cohesion"]
    )

    summarize(
        "ALIGNMENT — expected toward relative velocity",
        results["alignment"]
    )

    summarize(
        "PREDATOR AVOIDANCE — expected away from predator",
        results["predator_avoid"]
    )

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()