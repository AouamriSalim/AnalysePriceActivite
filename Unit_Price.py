import pandas as pd

# Load your dataset
df = pd.read_csv('Price_OS.csv')

# Explore the dataset
print(df.info())
print(df.head())

unique_activites = df['ACTIVITE'].unique()
print(unique_activites)

# Example: Simple random sample for each activite
sample_size = 30
random_samples = df.groupby('ACTIVITE').apply(lambda x: x.sample(sample_size)).reset_index(drop=True)

descriptive_stats = random_samples.groupby('ACTIVITE')['prix_unitaire'].agg(['mean', 'median', lambda x: x.mode().iloc[0]]).reset_index()
descriptive_stats.columns = ['ACTIVITE', 'mean', 'median', 'mode']

#print(descriptive_stats)
