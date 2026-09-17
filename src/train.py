import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


parser = argparse.ArgumentParser(description="Train an Iris Decision Tree classifier.")
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--random-state", type=int, default=42)
args = parser.parse_args()


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=args.test_size,
    random_state=args.random_state
)

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=args.random_state)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Create the outputs folder programmatically
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Save the trained model
joblib.dump(model, output_dir / "model.joblib")

# Create and save the confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("Iris Classifier - Confusion Matrix")
plt.tight_layout()

output_path = output_dir / "confusion_matrix.png"
plt.savefig(output_path)
plt.close()

print(f"Confusion matrix saved to: {output_path}")