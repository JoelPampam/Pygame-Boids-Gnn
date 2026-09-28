import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import BoidsGraphDataset
from model import BoidsGraphNet


# ============================================================
# SETTINGS
# ============================================================

TRAIN_PATH = "Data/dataset_1/processed/boids_log_1.pt"
VAL_PATH = "Data/dataset_1/processed/boids_log_2.pt"

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

MODEL_SAVE_PATH = "boids_gnn_v2.pt"


# ============================================================
# CREATE BATCH
# ============================================================

def create_batch(dataset, start, end):
    """
    Combine multiple timestep graphs into one large
    disconnected graph.

    Example with batch size 32:

        32 * 37 = 1184 nodes
        32 * 1332 = 42624 edges

    Graphs do NOT connect to each other.
    """

    graphs = [
        dataset[i]
        for i in range(start, end)
    ]

    x_list = []
    edge_index_list = []
    edge_attr_list = []

    y_list = []
    boid_mask_list = []

    node_offset = 0

    for graph in graphs:

        x_list.append(
            graph["x"]
        )

        # Shift node indices so this graph occupies
        # its own section of the large batch graph.
        edge_index_list.append(
            graph["edge_index"] + node_offset
        )

        edge_attr_list.append(
            graph["edge_attr"]
        )

        y_list.append(
            graph["y"]
        )

        boid_mask_list.append(
            graph["boid_mask"]
        )

        node_offset += graph["x"].shape[0]

    batch = {
        "x": torch.cat(
            x_list,
            dim=0
        ),

        "edge_index": torch.cat(
            edge_index_list,
            dim=1
        ),

        "edge_attr": torch.cat(
            edge_attr_list,
            dim=0
        ),

        "y": torch.cat(
            y_list,
            dim=0
        ),

        "boid_mask": torch.cat(
            boid_mask_list,
            dim=0
        )
    }

    return batch


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    dataset,
    optimizer,
    loss_function
):

    model.train()

    total_loss = 0.0
    num_batches = 0

    for start in range(
        0,
        len(dataset),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(dataset)
        )

        batch = create_batch(
            dataset,
            start,
            end
        )

        optimizer.zero_grad()

        predicted_delta_v, interactions = model(
            batch["x"],
            batch["edge_index"],
            batch["edge_attr"]
        )

        mask = batch["boid_mask"]

        prediction = predicted_delta_v[mask]
        target = batch["y"][mask]

        loss = loss_function(
            prediction,
            target
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    dataset,
    loss_function
):

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        for start in range(
            0,
            len(dataset),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(dataset)
            )

            batch = create_batch(
                dataset,
                start,
                end
            )

            predicted_delta_v, interactions = model(
                batch["x"],
                batch["edge_index"],
                batch["edge_attr"]
            )

            mask = batch["boid_mask"]

            prediction = predicted_delta_v[mask]
            target = batch["y"][mask]

            loss = loss_function(
                prediction,
                target
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BOIDS GRAPHNET BATCHED TRAINING")
    print("=" * 60)

    print()
    print("Loading training data...")

    train_dataset = BoidsGraphDataset(
        TRAIN_PATH
    )

    print()
    print("Loading validation data...")

    val_dataset = BoidsGraphDataset(
        VAL_PATH
    )

    model = BoidsGraphNet()

    loss_function = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print()
    print(f"Batch size: {BATCH_SIZE}")

    train_batches = (
        len(train_dataset)
        + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"Training batches per epoch: "
        f"{train_batches}"
    )

    print()

    # ========================================================
    # TRAIN
    # ========================================================

    best_val_loss = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_epoch(
            model,
            train_dataset,
            optimizer,
            loss_function
        )

        val_loss = validate(
            model,
            val_dataset,
            loss_function
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print(
                f"  -> Saved best model "
                f"(val loss = "
                f"{best_val_loss:.6f})"
            )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best model saved to: "
        f"{MODEL_SAVE_PATH}"
    )


if __name__ == "__main__":
    main()