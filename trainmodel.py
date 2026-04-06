import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# 1. Veri yükleme
# -----------------------------
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1).values
y = df["target"].values.reshape(-1, 1)
feature_names = df.drop("target", axis=1).columns

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. Normalization
# -----------------------------
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_std[X_train_std == 0] = 1

X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std
np.save("x_train_mean.npy", X_train_mean)
np.save("x_train_std.npy", X_train_std)

# -----------------------------
# 3. Sıfırdan model
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(z)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            if epoch % 1000 == 0:
                print(f"MyModel Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


# -----------------------------
# 4. Hazır modeller
# -----------------------------
# Sklearn Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train.ravel())
lr_preds = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)

# Random Forest
rf = RandomForestClassifier(
    max_depth=5,
    n_estimators=100,
    random_state=42
)
rf.fit(X_train_scaled, y_train.ravel())
rf_preds = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)


# -----------------------------
# 5. Sıfırdan model
# -----------------------------
my_model = MyLogisticRegression(learning_rate=0.01, epochs=10000)
my_model.fit(X_train_scaled, y_train)

my_preds = my_model.predict(X_test_scaled)
my_acc = accuracy_score(y_test, my_preds)

print("Final loss:", my_model.loss_history[-1])


# -----------------------------
# 6. Sonuçlar
# -----------------------------
print("\n--- Model Results ---")
print(f"Sklearn Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"My Logistic Regression Accuracy: {my_acc:.4f}")


# -----------------------------
# 7. Grafikler
# -----------------------------

# 7.1 Model Comparison
models = ["Sklearn Logistic", "Random Forest", "My Logistic"]
scores = [lr_acc, rf_acc, my_acc]

plt.figure(figsize=(8, 5))
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# 7.2 Feature Importance (Random Forest)
importances = rf.feature_importances_

plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# 7.3 Loss Graph (My Model)
plt.figure(figsize=(8, 5))
plt.plot(my_model.loss_history)
plt.title("Training Loss of My Logistic Regression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

print("Saved plots: model_comparison.png, feature_importance.png, loss_curve.png")


# -----------------------------
# 8. En iyi modeli seçme
# -----------------------------
results = {
    "sklearn_logistic": lr_acc,
    "random_forest": rf_acc,
    "my_logistic": my_acc
}

best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name}")


# -----------------------------
# 9. Kaydetme
# -----------------------------
if best_model_name == "sklearn_logistic":
    joblib.dump(lr, "best_model.pkl")
    print("Saved: best_model.pkl (sklearn logistic)")

elif best_model_name == "random_forest":
    joblib.dump(rf, "best_model.pkl")
    print("Saved: best_model.pkl (random forest)")

else:
    np.save("my_model_weights.npy", my_model.weights)
    np.save("my_model_bias.npy", np.array([my_model.bias]))
    print("Saved: my_model_weights.npy and my_model_bias.npy")