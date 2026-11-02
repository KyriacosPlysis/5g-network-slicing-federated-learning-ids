import pandas as pd  # For loading CSVs
from sklearn.model_selection import train_test_split  # For train/val splitting
from sklearn.preprocessing import StandardScaler  # For feature scaling
from tensorflow.keras.models import Sequential  # type: ignore # For building the neural network
from tensorflow.keras.layers import Dense, Dropout  # type: ignore # Neural network layers
from tensorflow.keras.optimizers import Adam  # type: ignore # Optimizer for training
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore # Prevents overfitting
import matplotlib.pyplot as plt  # Optional: plot training history

# ----------------------------
# Load and preprocess dataset
# ----------------------------
def load_data(path):
    df = pd.read_csv(path)  # Load the CSV file into a DataFrame

    X = df.drop(columns=["Label"])  # Features
    y = df["Label"]  # Target variable

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, scaler

# ----------------------------
# Build the neural network model
# ----------------------------
def build_model(input_dim):
    model = Sequential([
        Dense(16, input_dim=input_dim, activation="relu"),
        Dropout(0.3),
        Dense(8, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")  # Output layer for binary classification
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ----------------------------
# Early stopping callback
# ----------------------------
def get_early_stopping():
    return EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

# ----------------------------
# Optional: plot training history
# ----------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
