import os
import flwr as fl
from utils import build_model, load_data

# Create directory to store all model weights per round
os.makedirs("saved_models/tune_fedavgm", exist_ok=True)

# Function to run a single experiment with given hyperparams
def run_experiment(learning_rate, momentum):
    print(f"\n🚀 Running FedAvgM with LR={learning_rate}, Momentum={momentum}")

    accuracy_history = []

    # Define custom evaluation function for server
    def get_eval_fn():
        X_train, X_val, y_train, y_val, scaler = load_data("FL_5GSlices_subsets/global_val_set.csv")
        model = build_model(input_dim=X_val.shape[1])

        def evaluate(server_round, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            accuracy_history.append((server_round, accuracy))

            # Save weights
            filename = f"saved_models/tune_fedavgm/round_{server_round}_acc_{accuracy * 100:.2f}.weights.h5"
            model.save_weights(filename)
            print(f"💾 Saved weights: {filename}")

            return loss, {"accuracy": accuracy}

        return evaluate

    # Load model for initial weights
    _, X_val, _, y_val, _ = load_data("FL_5GSlices_subsets/global_val_set.csv")
    model = build_model(input_dim=X_val.shape[1])
    # Uncomment the next line to use a specific pre-trained checkpoint
    # model.load_weights("aved_models/fedavgm_models/Slices/init_Slices_acc_88.28.weights.h5")
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

    # Build strategy
    strategy = fl.server.strategy.FedAvgM(
        min_available_clients=3,
        evaluate_fn=get_eval_fn(),
        server_learning_rate=learning_rate,
        server_momentum=momentum,
        initial_parameters=initial_parameters
    )

    # Start server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy
    )

    final_acc = accuracy_history[-1][1] if accuracy_history else 0.0
    print(f"\n📈 Final Accuracy for LR={learning_rate}, Momentum={momentum}: {final_acc*100:.2f}%")

    return (learning_rate, momentum, final_acc)


# Grid Search Parameters
learning_rates = [0.3, 0.5, 1.0]
momentums = [0.9, 0.3, 0.6, 0.8]
results = []

# Grid Search Loop
for lr in learning_rates:
    for mom in momentums:
        input(f"\n⚡ Start your clients and press ENTER to run server with LR={lr}, Momentum={mom}...")
        result = run_experiment(lr, mom)
        results.append(result)

# Print Final Summary
print("\n🧪 Grid Search Summary:")
for lr, mom, acc in results:
    print(f"  LR={lr:<4} | Momentum={mom:<4} → Accuracy: {acc*100:.2f}%")

best = max(results, key=lambda x: x[2])
print(f"\n🏆 Best Combo → LR={best[0]}, Momentum={best[1]} with Accuracy={best[2]*100:.2f}%")
