import torch

from dataset import BoidsGraphDataset
from model import BoidsGraphNet


MODEL_PATH = "boids_gnn_v2.pt"
DATA_PATH = "Data/dataset_1/processed/boids_log_2.pt"


# ============================================================
# HELPER
# ============================================================

def get_rule_ids(ground_truth, timestep, entity_id, rule_name):
    """
    Get the simulator's known neighbor IDs for one boid
    at one timestep.

    These labels are ONLY used for analysis.
    They were never provided to the GNN as inputs.
    """

    frame = ground_truth[timestep]

    # ground_truth is stored as one entry per dynamic entity.
    for entity in frame:

        if (
            entity["entity_type"] == "boid"
            and entity["entity_id"] == entity_id
        ):
            return entity.get(rule_name, [])

    return []


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 60)
    print("LEARNED INTERACTION ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Load validation trajectory
    # --------------------------------------------------------

    dataset = BoidsGraphDataset(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = BoidsGraphNet()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            weights_only=True
        )
    )

    model.eval()

    print()
    print("Loaded trained model.")
    print("Analyzing validation trajectory...")
    print()

    # --------------------------------------------------------
    # Rules we want to investigate
    # --------------------------------------------------------

    rule_names = [
        "separation",
        "cohesion",
        "alignment",
        "predator_avoid"
    ]

    results = {}

    for rule in rule_names:

        results[rule] = {
            "active": [],
            "inactive": []
        }

    # --------------------------------------------------------
    # Analyze every validation timestep
    # --------------------------------------------------------

    with torch.no_grad():

        for timestep in range(len(dataset)):

            graph = dataset[timestep]

            prediction, interactions = model(
                graph["x"],
                graph["edge_index"],
                graph["edge_attr"]
            )

            senders = graph["edge_index"][0]
            receivers = graph["edge_index"][1]

            entity_types = graph["entity_types"]
            entity_ids = graph["entity_ids"]

            # ------------------------------------------------
            # Interaction magnitude
            #
            # ||m_(j->i)||
            # ------------------------------------------------

            magnitudes = torch.linalg.vector_norm(
                interactions,
                dim=1
            )

            # ------------------------------------------------
            # Inspect every edge
            # ------------------------------------------------

            for edge_index in range(
                interactions.shape[0]
            ):

                sender_index = int(
                    senders[edge_index]
                )

                receiver_index = int(
                    receivers[edge_index]
                )

                # We are currently analyzing effects ON boids.
                if int(entity_types[receiver_index]) != 0:
                    continue

                receiver_id = int(
                    entity_ids[receiver_index]
                )

                sender_id = int(
                    entity_ids[sender_index]
                )

                sender_type = int(
                    entity_types[sender_index]
                )

                magnitude = float(
                    magnitudes[edge_index]
                )

                # ============================================
                # BOID -> BOID RULES
                # ============================================

                if sender_type == 0:

                    for rule in [
                        "separation",
                        "cohesion",
                        "alignment"
                    ]:

                        active_ids = get_rule_ids(
                            dataset.run["ground_truth"],
                            timestep,
                            receiver_id,
                            rule
                        )

                        if sender_id in active_ids:

                            results[rule]["active"].append(
                                magnitude
                            )

                        else:

                            results[rule]["inactive"].append(
                                magnitude
                            )

                # ============================================
                # PREDATOR -> BOID
                # ============================================

                elif sender_type == 1:

                    active_ids = get_rule_ids(
                        dataset.run["ground_truth"],
                        timestep,
                        receiver_id,
                        "predator_avoid"
                    )

                    if sender_id in active_ids:

                        results["predator_avoid"]["active"].append(
                            magnitude
                        )

                    else:

                        results["predator_avoid"]["inactive"].append(
                            magnitude
                        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    for rule in rule_names:

        active = results[rule]["active"]
        inactive = results[rule]["inactive"]

        print()
        print(rule.upper())
        print("-" * 60)

        print(
            f"Active edges:   {len(active)}"
        )

        print(
            f"Inactive edges: {len(inactive)}"
        )

        if len(active) > 0:

            active_tensor = torch.tensor(
                active
            )

            active_mean = (
                active_tensor.mean().item()
            )

            active_median = (
                active_tensor.median().item()
            )

            print(
                f"Active mean magnitude:   "
                f"{active_mean:.6f}"
            )

            print(
                f"Active median magnitude: "
                f"{active_median:.6f}"
            )

        else:

            active_mean = None

            print(
                "No active edges found."
            )

        if len(inactive) > 0:

            inactive_tensor = torch.tensor(
                inactive
            )

            inactive_mean = (
                inactive_tensor.mean().item()
            )

            inactive_median = (
                inactive_tensor.median().item()
            )

            print(
                f"Inactive mean magnitude:   "
                f"{inactive_mean:.6f}"
            )

            print(
                f"Inactive median magnitude: "
                f"{inactive_median:.6f}"
            )

        else:

            inactive_mean = None

        # ----------------------------------------------------
        # Ratio
        # ----------------------------------------------------

        if (
            active_mean is not None
            and inactive_mean is not None
            and inactive_mean > 0
        ):

            ratio = (
                active_mean /
                inactive_mean
            )

            print(
                f"Active / inactive ratio: "
                f"{ratio:.3f}x"
            )

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()