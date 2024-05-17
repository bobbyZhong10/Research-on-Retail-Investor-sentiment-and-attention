import matplotlib
matplotlib.use('TkAgg')  # or 'agg'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import shap

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

# AdaBoost model parameters for grid search
params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}
model = AdaBoostClassifier(algorithm='SAMME')

# Train the model using GridSearchCV with cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, params, cv=kf)
grid_search.fit(DVs, IV)
best_model = grid_search.best_estimator_  # Extract the best model

# Evaluate the best model
y_pred = best_model.predict(DVs)
y_pred_proba = best_model.predict_proba(DVs)[:, 1]
metrics = evaluate_model(IV, y_pred, y_pred_proba)
print("Model Performance Metrics:", metrics)

# Use KernelExplainer to compute SHAP values
background = shap.sample(DVs, 100)  # Use a small background sample to speed up calculations
explainer = shap.KernelExplainer(best_model.predict_proba, background)
shap_values = explainer.shap_values(DVs, nsamples=1000)  # Compute SHAP values for all DVs

# Print the structure of shap_values
print("SHAP values structure:", type(shap_values), shap_values.shape)

# Since shap_values is a 3D array, need to extract the correct class
shap_values_class1 = shap_values[:, :, 1]

# Verify shapes
print(f"Shape of DVs: {DVs.shape}")
print(f"Shape of shap_values_class1: {shap_values_class1.shape}")

# Feature names for labeling
feature_names = DVs.columns.tolist()

# Check shape alignment
if shap_values_class1.shape[0] != DVs.shape[0] or shap_values_class1.shape[1] != DVs.shape[1]:
    print("Mismatch in shapes between SHAP values and data matrix")
else:
    # Generate SHAP summary plot for the first class
    shap.summary_plot(shap_values_class1, DVs, feature_names=feature_names, plot_type="dot")

    # Save and show the summary plot
    plt.savefig('shap_summary_plot.png', bbox_inches='tight')
    plt.show()

# Generate SHAP Waterfall plot for a specific instance
instance_index = 0  # Change this index to choose different instances
shap_waterfall_values = explainer.shap_values(DVs.iloc[[instance_index]], nsamples=1000)

# Extract the SHAP values for the specific class
shap_waterfall_values_class1 = shap.Explanation(values=shap_waterfall_values[:, :, 1][0],
                                                base_values=explainer.expected_value[1],
                                                data=DVs.iloc[instance_index],
                                                feature_names=feature_names)

shap.plots.waterfall(shap_waterfall_values_class1, show=False)

# Adjust plot aesthetics and save the Waterfall plot
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['top'].set_visible(True)
plt.savefig('shap_waterfall_plot.png', bbox_inches='tight', dpi=400)
plt.show()

# Save SHAP values to CSV file
shap_values_df = pd.DataFrame(shap_values_class1, columns=feature_names)
shap_values_df.to_csv('SHAP.csv', index=False)
