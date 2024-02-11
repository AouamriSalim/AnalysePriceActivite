import pandas as pd

# Assuming 'data_for_estimation' is your DataFrame
# Replace this with your actual DataFrame
file_path = 'D:\Work Cos\clustred Price.csv'
data_for_estimation = pd.read_csv(file_path)
# Ensure 'prix_unitaire' is converted to float
data_for_estimation['prix_unitaire'] = data_for_estimation['prix_unitaire'].replace(',', '.', regex=True).astype(float)

# Calculate the mean prix_unitaire for each Code_AC1
estimated_prices = data_for_estimation.groupby('Code_AC1')['prix_unitaire'].mean().reset_index()

# Print the estimated prices
print(estimated_prices)

output_file_path = 'D:\Work Cos\estimated Price.xlsx'
#estimated_prices[['Code_AC1','prix_unitaire']].to_excel(output_file_path, index=False)
