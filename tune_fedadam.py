import os
import flwr as fl
from utils import build_model, load_data

# Create folder to store all model weights per round
os.makedirs("saved_models/fedadam_models", exist_ok=True)

# Function to run one experiment with specific hyperparams
def run_experiment(eta, beta_1, beta_2):
    print(f"\n🚀 Running FedAdam with η={eta}, β1={beta_1}, β2={beta_2}")

    accuracy_history = []

    # Define custom evaluation logic for the server
    def get_eval_fn():
        X_train, X_val, y_train, y_val, scaler = load_data("FL_5GSlices_subsets/global_val_set.csv")
        model = build_model(input_dim=X_val.shape[1])

        def evaluate(server_round, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
            accuracy_history.append((server_round, accuracy))

            # Save weights for every round (for manual inspection later)
            filename = f"saved_models/fedadam_models/round_{server_round}_acc_{accuracy * 100:.2f}.weights.h5"
            model.save_weights(filename)
            print(f"💾 Saved weights: {filename}")

            return loss, {"accuracy": accuracy}

        return evaluate

    # Initialize model with previously saved weights
    _, X_val, _, y_val, _ = load_data("FL_5GSlices_subsets/global_val_set.csv")
    model = build_model(input_dim=X_val.shape[1])

    # Uncomment the next line to use pre-trained initialization
    # model.load_weights("saved_models/fedadam_models/Slices/init_Slices_acc_89.98.weights.h5")
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

    # Define strategy
    strategy = fl.server.strategy.FedAdam(
        min_available_clients=3,
        evaluate_fn=get_eval_fn(),
        eta=eta,
        beta_1=beta_1,
        beta_2=beta_2,
        initial_parameters=initial_parameters,
    )

    # Start the FL server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy
    )

    # Return final accuracy after training
    final_acc = accuracy_history[-1][1] if accuracy_history else 0.0
    print(f"\n📈 Final Accuracy for η={eta}, β1={beta_1}, β2={beta_2}: {final_acc*100:.2f}%")
    return (eta, beta_1, beta_2, final_acc)

# Grid Search Parameters
etas = [0.09, 0.003]
betas_1 = [0.95]
betas_2 = [0.99]
results = []


# Grid Search Loop
for eta in etas:
    for b1 in betas_1:
        for b2 in betas_2:
            input(f"\n⚡ Start your clients and press ENTER to run server with η={eta}, β1={b1}, β2={b2}...")
            result = run_experiment(eta, b1, b2)
            results.append(result)


# Print Summary
print("\n🧪 Grid Search Summary:")
for eta, b1, b2, acc in results:
    print(f"  η={eta:<6} | β1={b1:<4} | β2={b2:<5} → Accuracy: {acc*100:.2f}%")

# Print the best configuration
best = max(results, key=lambda x: x[3])
print(f"\n🏆 Best → η={best[0]}, β1={best[1]}, β2={best[2]} with Accuracy={best[3]*100:.2f}%")

