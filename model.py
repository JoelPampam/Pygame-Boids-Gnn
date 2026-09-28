import torch
import torch.nn as nn


# ============================================================
# EDGE INTERACTION MODEL
# ============================================================

class EdgeModel(nn.Module):

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

        return self.network(inputs)


# ============================================================
# SELF DYNAMICS MODEL
# ============================================================

class SelfModel(nn.Module):
    """
    Models velocity changes that cannot be attributed to
    another dynamic entity.

    Examples may include:
        leader behavior
        waypoint following
        other external/self dynamics

    This prevents the edge network from being forced to
    explain every acceleration using another entity.
    """

    def __init__(
        self,
        node_dim=7,
        hidden_dim=32
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                node_dim,
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

    def forward(self, x):

        return self.network(x)


# ============================================================
# INTERPRETABLE BOIDS GRAPHNET
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

        self.self_model = SelfModel(
            node_dim=node_dim
        )


    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        # ----------------------------------------------------
        # Directed edges
        #
        # sender j -> receiver i
        # ----------------------------------------------------

        senders = edge_index[0]
        receivers = edge_index[1]

        sender_features = x[senders]
        receiver_features = x[receivers]

        # ----------------------------------------------------
        # DIRECT EDGE CONTRIBUTIONS
        #
        # These are now intended to directly represent
        # contributions to delta-v.
        # ----------------------------------------------------

        interactions = self.edge_model(
            receiver_features,
            sender_features,
            edge_attr
        )

        # ----------------------------------------------------
        # SUM INCOMING INTERACTIONS
        # ----------------------------------------------------

        num_nodes = x.shape[0]

        interaction_sum = torch.zeros(
            num_nodes,
            2,
            device=x.device,
            dtype=x.dtype
        )

        interaction_sum.index_add_(
            0,
            receivers,
            interactions
        )

        # ----------------------------------------------------
        # SELF / EXTERNAL DYNAMICS
        # ----------------------------------------------------

        self_effect = self.self_model(x)

        # ----------------------------------------------------
        # FINAL PREDICTION
        #
        # IMPORTANT:
        #
        # There is NO nonlinear NodeModel after aggregation.
        #
        # delta-v_i =
        #       self_effect_i
        #       +
        #       sum_j interaction_(j -> i)
        # ----------------------------------------------------

        delta_velocity = (
            self_effect
            +
            interaction_sum
        )

        return (
            delta_velocity,
            interactions
        )