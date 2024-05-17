import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('300_dataset.csv')
data['Wretwd_sign'] = data['Wretwd'].apply(lambda x: 1 if x > 0 else -1)

data['Date'] = pd.to_datetime(data['Date'])  # Ensure date format is correct
data['Week_Year'] = data['Date'].dt.strftime('%Y-%U')

# Define bad market weeks (90%, 80%, and 70% of stocks have negative returns)
def bad_market_weeks(percentage):
    negative_return_percent = data.groupby('Week_Year')['Wretwd_sign'].apply(lambda x: (x == -1).mean())
    return negative_return_percent[negative_return_percent > percentage].index

bad_weeks_90 = bad_market_weeks(0.9)
bad_weeks_80 = bad_market_weeks(0.8)
bad_weeks_70 = bad_market_weeks(0.7)

# Filter data for bad market weeks
bad_market_data_90 = data[data['Week_Year'].isin(bad_weeks_90)]
bad_market_data_80 = data[data['Week_Year'].isin(bad_weeks_80)]
bad_market_data_70 = data[data['Week_Year'].isin(bad_weeks_70)]

# Calculate correlations
correlations_90 = bad_market_data_90[['Wretwd_sign', 'SVI_All']].corr()
correlations_80 = bad_market_data_80[['Wretwd_sign', 'SVI_All']].corr()
correlations_70 = bad_market_data_70[['Wretwd_sign', 'SVI_All']].corr()

# Print correlations
print("Correlations when 90% of stocks have negative returns:\n", correlations_90)
print("Correlations when 80% of stocks have negative returns:\n", correlations_80)
print("Correlations when 70% of stocks have negative returns:\n", correlations_70)

# Plot and save correlation matrices
def plot_and_save_corr_matrix(corr_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

plot_and_save_corr_matrix(correlations_90, 'Correlation Matrix (90% Negative Returns)', 'correlations_90.png')
plot_and_save_corr_matrix(correlations_80, 'Correlation Matrix (80% Negative Returns)', 'correlations_80.png')
plot_and_save_corr_matrix(correlations_70, 'Correlation Matrix (70% Negative Returns)', 'correlations_70.png')
