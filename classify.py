import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming 'data_for_clustering' is your DataFrame
# Replace this with your actual DataFrame
file_path = 'D:\Work Cos\ReportPrixUnitMois.csv'
data_for_clustering = pd.read_csv(file_path)

# Handle NaN values in the 'Code_AC1' column
imputer = SimpleImputer(strategy='most_frequent')
data_for_clustering['Code_AC1'] = imputer.fit_transform(data_for_clustering['Code_AC1'].values.reshape(-1, 1)).flatten()

# Encode categorical columns using Label Encoding
label_encoder = LabelEncoder()
data_for_clustering['Code_AC1_encoded'] = label_encoder.fit_transform(data_for_clustering['Code_AC1'])
data_for_clustering['code_po_encoded'] = label_encoder.fit_transform(data_for_clustering['code_po'])

# Select relevant columns for clustering
columns_for_clustering = ['Code_AC1_encoded', 'code_po_encoded']

# Standardize the data
scaler = StandardScaler()
data_for_clustering_scaled = scaler.fit_transform(data_for_clustering[columns_for_clustering])

# Apply k-means clustering
kmeans = KMeans(n_clusters=20, random_state=42)  # Adjust the number of clusters as needed
data_for_clustering['cluster_by_Code_AC1'] = kmeans.fit_predict(data_for_clustering_scaled)

# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_for_clustering_scaled)

# Create a DataFrame for visualization
data_pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
data_pca_df['Cluster'] = data_for_clustering['cluster_by_Code_AC1']

# Visualize the data
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=data_pca_df, palette='viridis')
plt.title('K-Means Clustering (PCA)')
plt.show()
