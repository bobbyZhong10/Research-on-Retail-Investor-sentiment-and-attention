import pandas as pd

# Load data
data = pd.read_csv('300_dataset.csv')
data['Wretwd_sign'] = data['Wretwd'].apply(lambda x: 1 if x > 0 else -1)

# Calculate different statistics
stats = data[['Readnum', 'Commentnum', 'SVI_code', 'SVI_All']].agg(['mean', 'median', 'std', 'max', 'min'])

# Analyze reads and search index stats with negative wretwd_sign
negative_stats = data[data['Wretwd_sign'] == -1][['Readnum', 'Commentnum', 'SVI_code', 'SVI_All']].agg(['mean', 'median', 'std', 'max', 'min'])

print("Statistics (all data) :\n", stats)
print("Statistics (Wretwd_sign is negative) :\n", negative_stats)