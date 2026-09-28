import torch
import torch.nn as nn


# ============================================================
# EDGE NETWORK
# ============================================================

class EdgeModel(nn.Module):
    """
    Learns the interaction from sender j to receiver i.

    Input:
        receiver node features
        sender node features
        relative edge features

    Output:
        2D interaction vector:
        [interaction_x, interaction_y]
    """

    def __init__(
        self,
        node_dim=7,
        edge_dim=5,
        hidden_dim=64
    ):
        super().__init__()

        input_dim = (
            node_dim * 2
            + edge_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                2
            )
        )

    def forward(
        self,
        receiver_features,
        sender_features,
        edge_features
    ):

        inputs = torch.cat(
            [
                receiver_features,
                sender_features,
                edge_features
            ],
            dim=1
        )

        interaction = self.network(
            inputs
        )

        return interaction


# ============================================================
# NODE MODEL
# ============================================================

class NodeModel(nn.Module):
    """
    Uses the summed incoming interactions to predict
    the velocity change of each entity.
    """

    def __init__(
        self,
        node_dim=7,
        hidden_dim=64
    ):
        super().__init__()

        # Node features + 2D aggregated interaction
        input_dim = node_dim + 2

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                2
            )
        )

    def forward(
        self,
        node_features,
        aggregated_interactions
    ):

        inputs = torch.cat(
            [
                node_features,
                aggregated_interactions
            ],
            dim=1
        )

        delta_velocity = self.network(
            inputs
        )

        return delta_velocity


# ============================================================
# BOIDS GRAPH NETWORK
# ============================================================

class BoidsGraphNet(nn.Module):

    def __init__(
        self,
        node_dim=7,
        edge_dim=5,
        hidden_dim=64
    ):
        super().__init__()

        self.edge_model = EdgeModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim
        )

        self.node_model = NodeModel(
            node_dim=node_dim,
            hidden_dim=hidden_dim
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        # ----------------------------------------------------
        # Get sender / receiver indices
        # ----------------------------------------------------

        senders = edge_index[0]
        receivers = edge_index[1]

        # ----------------------------------------------------
        # Get node features for every edge
        # ----------------------------------------------------

        sender_features = x[senders]
        receiver_features = x[receivers]

        # ----------------------------------------------------
        # EDGE MESSAGE PASSING
        #
        # j -> i
        # ----------------------------------------------------

        interactions = self.edge_model(
            receiver_features,
            sender_features,
            edge_attr
        )

        # ----------------------------------------------------
        # AGGREGATE INCOMING INTERACTIONS
        #
        # M_i = sum_j m_(j -> i)
        # ----------------------------------------------------

        num_nodes = x.shape[0]

        aggregated = torch.zeros(
            num_nodes,
            2,
            device=x.device,
            dtype=x.dtype
        )

        aggregated.index_add_(
            0,
            receivers,
            interactions
        )

        # ----------------------------------------------------
        # NODE UPDATE
        # ----------------------------------------------------

        delta_velocity = self.node_model(
            x,
            aggregated
        )

        return delta_velocity, interactions


# ============================================================
# TEST
# ============================================================

def main():

    from dataset import BoidsGraphDataset

    # Load one processed simulation run
    dataset = BoidsGraphDataset(
        "Data/dataset_1/processed/boids_log_1.pt"
    )

    # Get the graph for timestep 0
    graph = dataset[0]

    # Create the model
    model = BoidsGraphNet()

    # Run one forward pass
    predicted_delta_v, interactions = model(
        graph["x"],
        graph["edge_index"],
        graph["edge_attr"]
    )

    print()
    print("=" * 60)
    print("MODEL TEST")
    print("=" * 60)

    print()
    print("Node input:")
    print(graph["x"].shape)

    print()
    print("Edges:")
    print(graph["edge_index"].shape)

    print()
    print("Edge attributes:")
    print(graph["edge_attr"].shape)

    print()
    print("Predicted delta velocity:")
    print(predicted_delta_v.shape)

    print()
    print("Learned edge interactions:")
    print(interactions.shape)

    print()
    print("First predicted delta-v:")
    print(predicted_delta_v[0])

    print()
    print("Actual delta-v:")
    print(graph["y"][0])

    print()
    print("First 5 learned interactions:")
    print(interactions[:5])

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()