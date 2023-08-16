import sys
import numpy as np
import csv
import joblib
import pickle

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso, Ridge

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from collections import Counter
from imblearn.over_sampling import SMOTE

# Load data from space-separated file and enhace the data using oversampling, if needed.
def load_data(file_path):
    data = np.loadtxt(file_path)
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]

    class_counts = Counter(y)
    print("Class distribution:")
    for label, count in class_counts.items():
        print(f"Class {int(label)}: {count} samples")

    max_samples = max(class_counts.values())
    min_samples = min(class_counts.values())
    threshold = 0.7

    if min_samples / max_samples < threshold:
        print("Class imbalance detected. Performing over-sampling with SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    else:
        return X, y

# Classifer definitions
def random_forest_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=25, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def xgboost_classifier(X_train, y_train):
    clf = XGBClassifier(learning_rate=0.3, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def gradient_boosting_classifier(X_train, y_train):
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def adaboost_classifier(X_train, y_train):
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def decision_tree_classifier(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def lasso_regression(X_train, y_train):
    clf = Lasso(alpha=0.1, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def ridge_regression(X_train, y_train):
    clf = Ridge(alpha=0.1, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Evaluate using different metrics
def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)  # Probabilities for log loss
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_pred_proba)
    return accuracy, precision, recall, f1, logloss


def write_model_metrics_to_csv(file_path, accuracy, precision, recall, f1, log_loss):
    with open(file_path, mode='w', newline='') as eval_file:
        eval_writer = csv.writer(eval_file)
        eval_writer.writerow(['Metric', 'Value'])
        eval_writer.writerow(['Accuracy', accuracy])
        eval_writer.writerow(['Precision', precision])
        eval_writer.writerow(['Recall', recall])
        eval_writer.writerow(['F1 Score', f1])
        eval_writer.writerow(['Log Loss', log_loss])

# Write feature importance for the final model to CSV file
def write_final_model_feature_importance_to_csv(clf_name, classifier):
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = classifier.feature_importances_
        with open(f'{clf_name}_final_model_feature_importance.csv', mode='w', newline='') as feature_file:
            feature_writer = csv.writer(feature_file)
            feature_writer.writerow(['Feature', 'Feature_Index', 'Importance'])
            for idx, importance in enumerate(feature_importances):
                feature_writer.writerow([f"Feature_{idx}", idx, importance])

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    X, y = load_data(data_file)

    # Classifiers
    classifiers = [
        ('Gradient_Boosting', gradient_boosting_classifier),
        ('Random_Forest', random_forest_classifier),
        ('XGBoost', xgboost_classifier),
        #('Decision_Tree', decision_tree_classifier),
        #('Adaboost', adaboost_classifier),
        # Add other classifiers here if needed
    ]

    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for clf_name, clf_func in classifiers:
        print(f"\n{clf_name.replace('_', ' ')} 10-fold Cross-Validation:")

        # Initialize lists to store metrics for each fold
        accuracy_list, precision_list, recall_list, f1_list, logloss_list = [], [], [], [], []
        feature_importance_folds = []  # Added this line to initialize the list

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier = clf_func(X_train, y_train)
            accuracy, precision, recall, f1, logloss = evaluate_classifier(classifier, X_test, y_test)

            # Append metrics to lists
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            logloss_list.append(logloss)

            # Print metrics of each fold
            print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Log_loss-score: {logloss:.4f}")
            #print(f"Fold {fold+1} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

            # Save evaluation metrics to separate file for each fold
            fold_file_path = f'{clf_name}_fold_{fold+1}_metrics.csv'
            write_model_metrics_to_csv(fold_file_path, accuracy, precision, recall, f1, logloss)

            # Write feature importance for each fold
            write_final_model_feature_importance_to_csv(f"{clf_name}_fold_{fold+1}", classifier)
            feature_importance_folds.append(classifier.feature_importances_)  # Added this line to store feature importances

        # Calculate and print mean metrics over all folds
        mean_accuracy = np.mean(accuracy_list)
        mean_precision = np.mean(precision_list)
        mean_recall = np.mean(recall_list)
        mean_f1 = np.mean(f1_list)
        mean_logloss = np.mean(logloss_list)
        print(f"\nMean Metrics over 10 folds - Accuracy: {mean_accuracy:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}, F1-score: {mean_f1:.4f}, Log_loss-score: {mean_logloss:.4f}")

        # Save the final model using joblib
        model_file_path = f'{clf_name}_model.joblib'
        joblib.dump(classifier, model_file_path)
        print(f"Final {clf_name.replace('_', ' ')} model saved to: {model_file_path}")

        # Save the final model using pickle
        pickle_file_path = f'{clf_name}_model.pkl'
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(classifier, pickle_file)
        print(f"Final {clf_name.replace('_', ' ')} model saved to: {pickle_file_path}")

        # Save feature importance for the final model to file (AVERAGE OVER ALL FOLDS)
        if len(feature_importance_folds) > 0:
            avg_feature_importance = np.mean(feature_importance_folds, axis=0)
            with open(f'{clf_name}_final_model_feature_importance.csv', mode='w', newline='') as feature_file:
                feature_writer = csv.writer(feature_file)
                feature_writer.writerow(['Feature', 'Feature_Index', 'Importance'])
                for idx, importance in enumerate(avg_feature_importance):
                    feature_writer.writerow([f"Feature_{idx}", idx, importance])

if __name__ == "__main__":
    main()
