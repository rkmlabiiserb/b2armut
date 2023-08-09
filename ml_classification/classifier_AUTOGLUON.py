import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import sys
from collections import Counter
from imblearn.over_sampling import SMOTE

if len(sys.argv) != 2:
    print("Usage: python your_script_name.py path_to_data_file.txt")
    sys.exit(1)

data = np.loadtxt(sys.argv[1])
np.random.shuffle(data)  # Shuffling the data
X = data[:, :-1]  # Features (all columns except the last one)
y = data[:, -1]   # Labels (last column)

# Count the occurrences of each class
class_counts = Counter(y)
print("Class distribution:")
for label, count in class_counts.items():
    print(f"Class {int(label)}: {count} samples")

# Check for class imbalance and perform over-sampling with SMOTE if needed
max_samples = max(class_counts.values())
min_samples = min(class_counts.values())
threshold = 0.7  # You can adjust this threshold based on your data

if min_samples / max_samples < threshold:
        print("Class imbalance detected. Performing over-sampling with SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X = X_resampled
        y = y_resampled

# Prepare the data as a Pandas DataFrame
data_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
data_df["target"] = y

# 10-fold cross-validation
num_folds = 10

# Lists to store performance metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
logloss_scores = []

for fold in range(num_folds):
    # Split data into training and validation sets for this fold
    train_data = data_df.sample(frac=0.9, random_state=fold)
    val_data = data_df.drop(train_data.index)

    # Instantiate the AutoGluon TabularPredictor with the 'path' argument
    auto_ml_task = TabularPredictor(path=f"fold_{fold}/", label="target")

    # Fit the TabularPredictor on the training data and specify the output_directory
    auto_ml_task.fit(train_data=train_data,  ag_args_fit={'num_gpus': 1})

    # Evaluate performance on the validation set
    y_true = val_data["target"]
    y_pred = auto_ml_task.predict(val_data)
    accuracy_scores.append(accuracy_score(y_true, y_pred))
    precision_scores.append(precision_score(y_true, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_true, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_true, y_pred, average='weighted'))

    logloss = log_loss(y_true, auto_ml_task.predict_proba(val_data))
    logloss_scores.append(logloss)

    # Get feature importance for this fold using permutation importance
    feature_importance_fold = auto_ml_task.feature_importance(val_data, feature_stage='original', subsample_size=5000, num_shuffle_sets=10)

    # Save permutation importance to a CSV file for this fold
    feature_importance_fold.to_csv(f"permutation_importance_fold_{fold}.csv")

# Save performance metrics for each fold to separate files
np.savetxt("accuracy_scores.csv", accuracy_scores, delimiter=",")
np.savetxt("precision_scores.csv", precision_scores, delimiter=",")
np.savetxt("recall_scores.csv", recall_scores, delimiter=",")
np.savetxt("f1_scores.csv", f1_scores, delimiter=",")
np.savetxt("logloss_scores.csv", logloss_scores, delimiter=",")
