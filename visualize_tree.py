# test code to visualize a decision tree from a RandomForest model that was used to classify handwritten digits

import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model
model, target_names = joblib.load("digit_model.pkl")

# Select a single decision tree from the forest
tree = model.estimators_[0]  # pick the first tree

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    max_depth=3,         # limit depth for readability
    feature_names=[f'pixel_{i}' for i in range(64)],
    class_names=[str(i) for i in target_names],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree from RandomForest (Depth 3)")
plt.tight_layout()

plt.savefig("tree_visualization.png")
print("Saved tree to tree_visualization.png")

plt.show()

