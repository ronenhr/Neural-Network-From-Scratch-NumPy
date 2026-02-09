import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Data Loading and Helper Functions
# ==========================================

def load_data(train_path='train.csv', test_path='test.csv'):
    """
    Loads Fashion-MNIST data from CSV files.
    Handles potential header detection automatically.
    """
    # Detect header in train.csv
    df_check = pd.read_csv(train_path, nrows=5, header=None)
    try:
        float(df_check.iloc[0, 0])
        has_header = False
    except ValueError:
        has_header = True
        print("Header detected in Train CSV. Skipping first row.")

    header_setting = 0 if has_header else None
    train_df = pd.read_csv(train_path, header=header_setting)
    
    # Detect header in test.csv
    df_test_check = pd.read_csv(test_path, nrows=5, header=None)
    try:
        float(df_test_check.iloc[0, 0])
        test_header = None
    except ValueError:
        test_header = 0
    
    test_df = pd.read_csv(test_path, header=test_header)

    train_data = train_df.values
    
    # Split Labels and Features
    X_train = train_data[:, 1:].astype(np.float32) 
    y_train = train_data[:, 0].astype(int)
    
    X_test = test_df.values.astype(np.float32)
    
    print(f"Data Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test

def visualize_data(X, y, classes_map):
    """Visualizes a grid of samples from each class."""
    print("Displaying data sample... (Close window to continue)")
    fig, axes = plt.subplots(10, 4, figsize=(8, 20))
    fig.tight_layout()

    for class_id in range(10):
        class_indices = np.where(y == class_id)[0]
        selected_indices = class_indices[:4]

        for col, idx in enumerate(selected_indices):
            img = X[idx].reshape(28, 28)
            ax = axes[class_id, col]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if col == 0:
                ax.set_title(f"{classes_map[class_id]}", loc='left')
    plt.show()

# ==========================================
# Preprocessing
# ==========================================

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def train_val_split(X, y, val_ratio=0.2):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split_idx = int(num_samples * (1 - val_ratio))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def normalize_data(X_train, X_val, X_test):
    max_val = 255.0
    return X_train / max_val, X_val / max_val, X_test / max_val

# ==========================================
# Models
# ==========================================

class LogisticRegression:
    def __init__(self, input_dim, num_classes, lr=0.01, reg=0.0):
        self.W = np.random.randn(input_dim, num_classes) * 0.01
        self.b = np.zeros(num_classes)
        self.lr = lr
        self.reg = reg 

    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return softmax(z)
    
    def compute_loss(self, y_true_onehot, y_pred):
        m = y_true_onehot.shape[0]
        epsilon = 1e-15
        loss = -np.sum(y_true_onehot * np.log(y_pred + epsilon)) / m
        l2_loss = (self.reg / 2) * np.sum(self.W ** 2)
        return loss + l2_loss
    
    def backward(self, X, y_true_onehot, y_pred):
        m = X.shape[0]
        dz = y_pred - y_true_onehot
        dw = (1 / m) * np.dot(X.T, dz) + self.reg * self.W
        db = (1 / m) * np.sum(dz, axis=0)
        self.W -= self.lr * dw
        self.b -= self.lr * db

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, num_classes, lr, reg):
        # He Initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_classes) * np.sqrt(2. / hidden_dim)
        self.b2 = np.zeros(num_classes)
        self.lr = lr
        self.reg = reg

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_deriv(self, z):
        return (z > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.h = self.relu(self.z1)
        self.z2 = np.dot(self.h, self.W2) + self.b2
        self.probs = softmax(self.z2)
        return self.probs
    
    def compute_loss(self, y_true_onehot, y_pred):
        m = y_true_onehot.shape[0]
        epsilon = 1e-15
        loss = -np.sum(y_true_onehot * np.log(y_pred + epsilon)) / m
        l2_loss = (self.reg / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return loss + l2_loss
    
    def backward(self, X, y_true_onehot):
        m = X.shape[0]
        # Output Layer Gradients
        dz2 = self.probs - y_true_onehot
        dW2 = (1/m) * np.dot(self.h.T, dz2) + self.reg * self.W2
        db2 = (1/m) * np.sum(dz2, axis=0)

        # Hidden Layer Gradients
        dh = np.dot(dz2, self.W2.T)
        dz1 = dh * self.relu_deriv(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1) + self.reg * self.W1
        db1 = (1/m) * np.sum(dz1, axis=0)
        
        # Update Weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ==========================================
# Training Loop & Main Execution
# ==========================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=128):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    y_train_onehot = one_hot_encode(y_train, 10)
    y_val_onehot = one_hot_encode(y_val, 10)
    num_samples = X_train.shape[0]

    print(f"\nTraining {model.__class__.__name__}...")

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_onehot[indices]

        # Mini-batch Gradient Descent
        for i in range(0, num_samples, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            probs = model.forward(X_batch)
            
            if isinstance(model, LogisticRegression):
                model.backward(X_batch, y_batch, probs)
            else:
                model.backward(X_batch, y_batch)

        # Evaluation
        train_probs = model.forward(X_train)
        train_loss = model.compute_loss(y_train_onehot, train_probs)
        train_acc = np.mean(np.argmax(train_probs, axis=1) == y_train)

        val_probs = model.forward(X_val)
        val_loss = model.compute_loss(y_val_onehot, val_probs)
        val_acc = np.mean(np.argmax(val_probs, axis=1) == y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
    return train_losses, val_losses, train_accs, val_accs

def plot_history(history, title):
    train_loss, val_loss, train_acc, val_acc = history
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, val_loss, 'r', label='Val Loss')
    plt.title(f"{title} - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b', label='Train Acc')
    plt.plot(epochs, val_acc, 'r', label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load Data
    try:
        X_full, y_full, X_test = load_data()
    except FileNotFoundError:
        print("Error: train.csv or test.csv not found. Please download them from Kaggle.")
        return

    # 2. Visualize (Optional)
    classes = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
               5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
    # visualize_data(X_full, y_full, classes) # Uncomment to see images

    # 3. Preprocess
    print("Preprocessing data...")
    X_train, y_train, X_val, y_val = train_val_split(X_full, y_full, val_ratio=0.2)
    X_train, X_val, X_test_norm = normalize_data(X_train, X_val, X_test)
    print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

    # 4. Train Logistic Regression
    lr_model = LogisticRegression(input_dim=784, num_classes=10, lr=0.1, reg=0.0001)
    lr_history = train_model(lr_model, X_train, y_train, X_val, y_val, epochs=20)
    # plot_history(lr_history, 'Logistic Regression')
    np.savetxt('lr_pred.csv', lr_model.predict(X_test_norm), fmt='%d')
    print("Saved lr_pred.csv")

    # 5. Train Neural Network
    nn_model = NeuralNetwork(input_dim=784, hidden_dim=128, num_classes=10, lr=0.1, reg=0.001)
    nn_history = train_model(nn_model, X_train, y_train, X_val, y_val, epochs=20)
    # plot_history(nn_history, "Neural Network")
    np.savetxt("NN_pred.csv", nn_model.predict(X_test_norm), fmt="%d")
    print("Saved NN_pred.csv")

if __name__ == "__main__":
    main()


