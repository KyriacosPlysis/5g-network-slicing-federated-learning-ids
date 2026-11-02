import sys
import flwr as fl
from utils import load_data, build_model, get_early_stopping

# Dataset selection from command line
if len(sys.argv) != 2:
    print("Usage: python mmtc_client.py [mmtc | split3]")
    sys.exit(1)

# Determine which dataset to load based on input argument
dataset_choice = sys.argv[1].lower()
if dataset_choice == "mmtc":
    path = "FL_5GSlices_subsets/mMTC_bpso_global.csv"
elif dataset_choice == "split3":
    path = "FL_EqSplit_subsets/split3.csv"
else:
    print("❌ Invalid dataset choice! Use 'mmtc' or 'split3'")
    sys.exit(1)

# Load and preprocess the selected dataset
X_train, X_val, y_train, y_val, scaler = load_data(path)

# Build the local neural network model
model = build_model(input_dim=X_train.shape[1])

# Print final validation accuracy summary per round
def print_local_summary(client_id: str, accuracy_history: list):
    print(f"\n📊 Final Accuracy Summary for [{client_id.upper()} Client]:")
    for round_num, acc in accuracy_history:
        print(f"  Round {round_num:2d}: {acc * 100:.2f}%")


# Define Flower client
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_name: str):
        self.client_name = client_name
        self.round_counter = 0
        self.local_accuracy_history = []

    # Send model weights to server
    def get_parameters(self, config):
        return model.get_weights()

    # Train model locally with server-sent weights
    def fit(self, parameters, config):
        self.round_counter += 1
        model.set_weights(parameters)

        # Log weight average before training
        before_mean = sum(w.mean() for w in model.get_weights()) / len(model.get_weights())
        print(f"📦 [{self.client_name.upper()} Client] - Round {self.round_counter}: Mean weight BEFORE training: {before_mean:.6f}")

        # Train on local data
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=60,
            batch_size=32,
            callbacks=[get_early_stopping()],
            verbose=0
        )

        after_mean = sum(w.mean() for w in model.get_weights()) / len(model.get_weights())
        print(f"📦 [{self.client_name.upper()} Client] - Round {self.round_counter}: Mean weight AFTER training: {after_mean:.6f}")

        acc = history.history["val_accuracy"][-1]
        print(f"📡 [{self.client_name.upper()} Client] - Round {self.round_counter}: Local Validation Accuracy = {acc*100:.2f}%")

        self.local_accuracy_history.append((self.round_counter, acc))

        return model.get_weights(), len(X_train), {}

    # Evaluate model locally on validation set
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

        if self.round_counter >= config.get("server_round", 0):
            print_local_summary(self.client_name, self.local_accuracy_history)

        return loss, len(X_val), {"accuracy": accuracy}


# Start the client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FLClient(dataset_choice)
)
