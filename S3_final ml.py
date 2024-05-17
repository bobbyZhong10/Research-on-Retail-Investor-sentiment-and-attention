import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

# Load the dataset
file_path = '300_dataset.csv'
data = pd.read_csv(file_path)

# Convert Wretwd to binary labels
data['Wretwd'] = data['Wretwd'].apply(lambda x: 1 if x > 0 else 0)

# Define independent variable (IV) and dependent variables (DV)
IV = data['Wretwd']
DVs = data[['Readnum', 'Commentnum', 'SVI_code', 'SVI_All']]


# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred, y_pred_proba):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Recall'] = recall_score(y_true, y_pred)
    metrics['F1'] = f1_score(y_true, y_pred)
    metrics['ROC-AUC'] = roc_auc_score(y_true, y_pred_proba)
    return metrics


# Define the models and their parameter grids for grid search
models = {
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    'AdaBoost': (AdaBoostClassifier(algorithm='SAMME'), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }),
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }),
    'GaussianNB': (GaussianNB(), {}),
    'MLP': (MLPClassifier(max_iter=500), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'learning_rate_init': [0.001, 0.01, 0.1]
    })
}


# Perform grid search to find the best parameters
def perform_grid_search(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, params, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


results = []

for DV_name in DVs.columns:
    X = DVs[[DV_name]]
    y = IV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for model_name, (model, params) in models.items():
        best_model = perform_grid_search(model, params, X_train, y_train)
        kf = KFold(n_splits=5)

        fold_metrics = []

        for train_index, val_index in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

            best_model.fit(X_fold_train, y_fold_train)
            y_pred = best_model.predict(X_fold_val)
            y_pred_proba = best_model.predict_proba(X_fold_val)[:, 1]

            metrics = evaluate_model(y_fold_val, y_pred, y_pred_proba)
            fold_metrics.append(metrics)

        mean_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
        mean_metrics['Model'] = model_name
        mean_metrics['DV'] = DV_name
        results.append(mean_metrics)

# Convert results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('S3_evaluation_results.csv', index=False)
print("\nResults:")
print(results_df.to_string(index=False))

# Plot the evaluation metrics for each model and DV
colors = ['#929fff', '#ffa897', '#a565ef', '#ffa510', '#70ad47']
metrics = ['Accuracy', 'Recall', 'F1', 'ROC-AUC']

for metric in metrics:
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f'{metric} Comparison Across Models', fontsize=16)

    for i, DV_name in enumerate(DVs.columns):
        ax = axes[i // 2, i % 2]
        for j, model_name in enumerate(models.keys()):
            model_results = results_df[(results_df['Model'] == model_name) & (results_df['DV'] == DV_name)]
            if metric == 'ROC-AUC':
                ax.plot(model_results['Model'], model_results[metric], marker='o', linestyle='-', color=colors[j],
                        label=model_name)
            else:
                bars = ax.bar(model_results['Model'], model_results[metric], color=colors[j])
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3),
                            va='bottom')  # Display the value on top of the bar

        ax.set_title(DV_name)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        if metric == 'ROC-AUC' and i == 0:
            ax.legend()

    plt.savefig(f'{metric}.png')
    plt.close()
