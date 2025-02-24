# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
RANDOM_SEED = 42

# Task 1.1a: Downloading the data
print("Downloading penguin dataset...")
penguins = fetch_openml(name='penguins', parser="auto", as_frame=True).frame
print(f"Dataset shape: {penguins.shape}")

# Task 1.1b: Preparing the data
# 1. Remove rows with missing values
penguins_clean = penguins.dropna(axis=0)
print(f"Shape after removing NA values: {penguins_clean.shape}")

# 2. Select and separate features and target
# Target is 'species', inputs are the measurement columns
target = penguins_clean['species']
features = penguins_clean[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# Print some information about our data
print("\nFeatures:")
print(features.describe())
print("\nTarget distribution:")
print(target.value_counts())

# 3. Split the dataset into training and test sets
# Using stratify=target ensures that the class distribution is preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.4, random_state=RANDOM_SEED, stratify=target
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Task 1.2a: Training a Decision Tree Classifier
print("\nTraining Decision Tree Classifier...")
dt_classifier = DecisionTreeClassifier(random_state=RANDOM_SEED)
dt_classifier.fit(X_train, y_train)

# Task 1.2b: Testing the Classifier
print("Testing classifier...")
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Task 1.2d: Visualization
print("\nExporting decision tree visualization...")
feature_names = features.columns.tolist()
class_names = np.unique(target).tolist()

# Export tree to dot file
export_graphviz(
    dt_classifier,
    out_file="penguin_tree.dot",
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True
)

print("Decision tree exported to 'penguin_tree.dot'")
print("You can convert it to an image using: dot -Tpng penguin_tree.dot -o penguin_tree.png")
print("Or use an online converter: https://dreampuf.github.io/GraphvizOnline/")

# For convenience, let's print the first few levels of the tree in text format
def print_tree_text(tree, feature_names, class_names, max_depth=3):
    """Print a simple text representation of the decision tree"""
    tree_ = tree.tree_
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != -2:  # not a leaf
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            print(f"{indent}if {name} <= {threshold:.2f}:")
            if depth < max_depth:
                recurse(tree_.children_left[node], depth + 1)
            else:
                print(f"{indent}  [... tree continues ...]")
            print(f"{indent}else:  # {name} > {threshold:.2f}")
            if depth < max_depth:
                recurse(tree_.children_right[node], depth + 1)
            else:
                print(f"{indent}  [... tree continues ...]")
        else:  # leaf
            class_counts = tree_.value[node][0]
            class_idx = np.argmax(class_counts)
            class_name = class_names[class_idx]
            total = sum(class_counts)
            probs = [count/total for count in class_counts]
            prob_str = ", ".join([f"{class_names[i]}: {p:.2f}" for i, p in enumerate(probs)])
            print(f"{indent}return {class_name} ({prob_str})")
    
    print("\nSimplified Tree Structure (first few levels):")
    recurse(0, 0)

print_tree_text(dt_classifier, feature_names, class_names)

# Optional: Multiple runs to demonstrate randomness (Task 1.2c)
if False:  # Set to True to run this section
    print("\nDemonstrating randomness with multiple runs...")
    accuracies = []
    
    for i in range(5):
        # Split without fixing random state
        X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
            features, target, test_size=0.4
        )
        
        # Train without fixing random state
        dt_random = DecisionTreeClassifier()
        dt_random.fit(X_train_random, y_train_random)
        
        # Test
        y_pred_random = dt_random.predict(X_test_random)
        acc = accuracy_score(y_test_random, y_pred_random)
        accuracies.append(acc)
        print(f"Run {i+1} accuracy: {acc:.4f}")
    
    print(f"Mean accuracy: {np.mean(accuracies):.4f}")
    print(f"Standard deviation: {np.std(accuracies):.4f}")