import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '300_dataset.csv'
data = pd.read_csv(file_path)

# Set the style for the plots
sns.set(style="whitegrid")

# Create a pairplot to visualize the relationships between Readnum, Commentnum, SVI_code, and SVI_All
plt.figure(figsize=(12, 10))
sns.pairplot(data[['Readnum', 'Commentnum', 'SVI_code', 'SVI_All']])
plt.suptitle('Pairplot of Readnum, Commentnum, SVI_code, and SVI_All', y=1.02)
plt.savefig('dv.png')
plt.show()
