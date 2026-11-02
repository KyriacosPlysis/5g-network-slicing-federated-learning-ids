import sys
import os
import flwr as fl
import numpy as np
from utils import build_model, load_data

# Keep track of validation accuracy per round for summary at the end
accuracy_history = []

# Create folder for saved models
os.makedirs("saved_models", exist_ok=True)

# Define evaluation function with optional early stopping
def get_eval_fn(patience=10, enable_early_stopping=False):
    X_train, X_val, y_train, y_val, scaler = load_data("FL_5GSlices_subsets/global_val_set.csv")
    model = build_model(input_dim=X_val.shape[1])

    best_acc = 0.0
    best_round = -1
    rounds_since_improvement = 0

    def evaluate(server_round, parameters, config):
        nonlocal best_acc, best_round, rounds_since_improvement

        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        accuracy_history.append((server_round, accuracy))

        print(f"✅ [Server] Round {server_round}: Accuracy = {accuracy * 100:.2f}%")

        # Save model if it achieves a new best accuracy
        if accuracy > best_acc:
            best_acc = accuracy
            best_round = server_round
            rounds_since_improvement = 0
            filename = f"saved_models/server_models/round_{server_round}_acc_{accuracy * 100:.2f}.weights.h5"
            model.save_weights(filename)
            print(f"💾 Saved model weights: {filename}")

        else:
            rounds_since_improvement += 1

        # Trigger early stopping if no improvement for 'patience' rounds
        if enable_early_stopping and rounds_since_improvement >= patience:
            print(f"🛑 Early stopping triggered at round {server_round} (Best Acc: {best_acc * 100:.2f}%)")
            os._exit(0)

        return loss, {"accuracy": accuracy}

    return evaluate

# Select federated strategy and initialize model
def get_strategy(name):
    # Enable early stopping ONLY for fedadam
    if name == "fedadam":
        eval_fn = get_eval_fn(patience=10, enable_early_stopping=True)
    else:
        eval_fn = get_eval_fn(patience=10, enable_early_stopping=False)

    # Get initial parameters
    _, X_val, _, y_val, _ = load_data("FL_5GSlices_subsets/global_val_set.csv")
    model = build_model(input_dim=X_val.shape[1])

    # Load different initial weights depending on strategy and data (slices or splits)
    if name == "fedadam":
        # Load Weights for FedAdam Slices
        model.load_weights("saved_models/fedadam_models/Slices/init_Slices_acc_89.98.weights.h5")

        # # Load Weights for FedAdam Splits
        # model.load_weights("saved_models/fedadam_models/Splits/init_Splits_acc_92.07.weights.h5")

    elif name == "fedavgm":
        # Load Weights for FedAvgm Slices
        model.load_weights("saved_models/fedavgm_models/Slices/init_Slices_acc_88.28.weights.h5")

        # # Load Weights for FedAvgm Splits
        # model.load_weights("saved_models/fedavgm_models/Splits/init_Splits_acc_92.76.weights.h5")

    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

    # Choose strategy
    if name == "fedavg":
        return fl.server.strategy.FedAvg(
            min_available_clients=3,
            evaluate_fn=eval_fn
        )
    elif name == "fedavgm":
        return fl.server.strategy.FedAvgM(
            min_available_clients=3,
            evaluate_fn=eval_fn,
            server_learning_rate=0.03,
            server_momentum=0.9,
            initial_parameters=initial_parameters
        )
    elif name == "fedadam":
        return fl.server.strategy.FedAdam(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=3,
            evaluate_fn=eval_fn,
            eta=0.001,
            beta_1=0.95,
            beta_2=0.99,
            initial_parameters=initial_parameters
        )
    else:
        raise ValueError("❌ Invalid strategy name! Use: fedavg, fedavgm, fedadam")

# Entry point: parse strategy argument and start the server
def main():
    if len(sys.argv) != 2:
        print("Usage: python server.py [fedavg | fedavgm | fedadam]")
        sys.exit(1)

    strategy_name = sys.argv[1].lower()
    strategy = get_strategy(strategy_name)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=60),
        strategy=strategy,
    )

    print("\n📊 Final Accuracy Summary (Centralized Evaluation):")
    for round_num, acc in accuracy_history:
        print(f"  Round {round_num:2d}: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()