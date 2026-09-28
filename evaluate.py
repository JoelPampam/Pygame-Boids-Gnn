import torch
import torch.nn as nn

from dataset import BoidsGraphDataset
from model import BoidsGraphNet


MODEL_PATH = "boids_gnn.pt"

TRAIN_PATH = "Data/dataset_1/processed/boids_log_1.pt"
VAL_PATH = "Data/dataset_1/processed/boids_log_2.pt"


def evaluate_dataset(model, dataset):

    model.eval()

    total_model_loss = 0.0
    total_zero_loss = 0.0

    total_absolute_error = 0.0
    total_squared_error = 0.0

    total_values = 0
    total_graphs = 0

    mse = nn.MSELoss()

    with torch.no_grad():

        for graph in dataset:

            # -----------------------------------------------
            # Model prediction
            # -----------------------------------------------

            prediction, interactions = model(
                graph["x"],
                graph["edge_index"],
                graph["edge_attr"]
            )

            mask = graph["boid_mask"]

            prediction = prediction[mask]
            target = graph["y"][mask]

            # -----------------------------------------------
            # GNN MSE
            # -----------------------------------------------

            model_loss = mse(
                prediction,
                target
            )

            total_model_loss += model_loss.item()

            # -----------------------------------------------
            # Zero-change baseline
            #
            # Predict:
            #
            # delta-v = [0, 0]
            # -----------------------------------------------

            zero_prediction = torch.zeros_like(
                target
            )

            zero_loss = mse(
                zero_prediction,
                target
            )

            total_zero_loss += zero_loss.item()

            # -----------------------------------------------
            # MAE / RMSE
            # -----------------------------------------------

            error = prediction - target

            total_absolute_error += (
                torch.abs(error).sum().item()
            )

            total_squared_error += (
                (error ** 2).sum().item()
            )

            total_values += target.numel()

            total_graphs += 1

    model_mse = (
        total_model_loss /
        total_graphs
    )

    zero_mse = (
        total_zero_loss /
        total_graphs
    )

    mae = (
        total_absolute_error /
        total_values
    )

    rmse = (
        total_squared_error /
        total_values
    ) ** 0.5

    improvement = (
        (zero_mse - model_mse)
        / zero_mse
        * 100.0
    )

    return {
        "model_mse": model_mse,
        "zero_mse": zero_mse,
        "mae": mae,
        "rmse": rmse,
        "improvement": improvement
    }


def print_results(name, results):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"GNN MSE:            "
        f"{results['model_mse']:.6f}"
    )

    print(
        f"Zero-change MSE:    "
        f"{results['zero_mse']:.6f}"
    )

    print(
        f"MAE:                "
        f"{results['mae']:.6f}"
    )

    print(
        f"RMSE:               "
        f"{results['rmse']:.6f}"
    )

    print(
        f"Improvement vs zero: "
        f"{results['improvement']:.2f}%"
    )


def main():

    print("=" * 60)
    print("BOIDS GRAPHNET EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
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
    print(
        f"Loaded trained model: "
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_dataset = BoidsGraphDataset(
        TRAIN_PATH
    )

    val_dataset = BoidsGraphDataset(
        VAL_PATH
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    train_results = evaluate_dataset(
        model,
        train_dataset
    )

    val_results = evaluate_dataset(
        model,
        val_dataset
    )

    print_results(
        "TRAINING TRAJECTORY",
        train_results
    )

    print_results(
        "VALIDATION TRAJECTORY",
        val_results
    )


if __name__ == "__main__":
    main()