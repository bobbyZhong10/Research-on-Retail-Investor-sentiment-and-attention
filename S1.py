import pandas as pd
import scipy.stats as stats

# Load the dataset
file_path = '300_dataset.csv'
data = pd.read_csv(file_path)

# Add a new column 'IV' based on 'Wretwd'
data['IV'] = data['Wretwd'].apply(lambda x: 1 if x > 0 else 0)

# Define the dependent variables
dependent_vars = ['Readnum', 'Commentnum', 'SVI_code', 'SVI_All']

# Create a summary table to store t-test and variance test results
summary_table = []

for dv in dependent_vars:
    # Split data into groups based on 'IV'
    group_0 = data[data['IV'] == 0][dv]
    group_1 = data[data['IV'] == 1][dv]

    # Perform t-test
    t_test_result = stats.ttest_ind(group_0, group_1, equal_var=False)

    # Perform variance test (Levene's test)
    levene_test_result = stats.levene(group_0, group_1)

    # Append results to summary table
    summary_table.append({
        'Dependent Variable': dv,
        'T-test Statistic': t_test_result.statistic,
        'T-test p-value': t_test_result.pvalue,
        'ANOVA Statistic': levene_test_result.statistic,
        'ANOVA p-value': levene_test_result.pvalue
    })

# Convert summary table to DataFrame
summary_df = pd.DataFrame(summary_table)

# Save the DataFrame to a CSV file
output_file_path = 'S1_t+anova.csv'
summary_df.to_csv(output_file_path, index=False)

output_file_path
