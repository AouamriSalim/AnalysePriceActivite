import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data_for_clustering' is your DataFrame
# Replace this with your actual DataFrame
file_path = 'D:\Work Cos\clustred Price.csv'
data_for_clustering = pd.read_csv(file_path)
# Pair Plot
sns.pairplot(data_for_clustering[['prix_unitaire', 'cluster']], hue='cluster')
plt.title('Pair Plot')
plt.show()

# Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster', y='prix_unitaire', data=data_for_clustering)
plt.title('Box Plot')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='code_po', y='prix_unitaire', hue='cluster', data=data_for_clustering)
plt.title('Scatter Plot')
plt.show()

# Cluster Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='code_po', y='prix_unitaire', hue='cluster', data=data_for_clustering)
plt.title('Cluster Visualization')
plt.show()
