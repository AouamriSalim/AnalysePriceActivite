
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pyswarm import pso
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Assuming your dataset is stored in a DataFrame named df

df = pd.read_csv('Price_all.csv')

print(df.head())
# Convert relevant columns to numeric
df['prix_unitaire'] = pd.to_numeric(df['prix_unitaire'].str.replace(',', '.'), errors='coerce')
df['quantite'] = pd.to_numeric(df['quantite'].str.replace(',', '.'), errors='coerce')
df['Code_AC'] = pd.to_numeric(df['Code_AC'], errors='coerce')

# Define the objective function for PSO
def objective_function(params, X_ac, y_ac):
    prix_unitaire = params[0]
    df_filtered = df[df['Code_AC'].isin(X_ac['Code_AC'])]  # Use .isin() for proper comparison
    X_filtered = df_filtered[['quantite']]
    y_filtered = df_filtered['prix_unitaire']

    # Drop rows with NaN values in either y_filtered or y_pred
    non_nan_indices = ~np.isnan(y_filtered) & ~np.isnan(np.full_like(y_filtered, prix_unitaire))
    y_filtered = y_filtered[non_nan_indices]
    y_pred = np.full_like(y_filtered, prix_unitaire)  # Assuming constant prix_unitaire for each Code_AC

    mse = mean_squared_error(y_filtered, y_pred)
    return mse


# Split the data into training and testing sets
X_train, X_test, _, _ = train_test_split(df[['Code_AC']], df['prix_unitaire'], test_size=0.2, random_state=42)

# Run PSO optimization for each unique Code_AC
optimized_params = []

for code_ac in X_train['Code_AC'].unique():
    X_ac = X_train[X_train['Code_AC'] == code_ac]
    y_ac = df[df['Code_AC'] == code_ac]['prix_unitaire']

    lb = [y_ac.min()]  # Lower bound for prix_unitaire
    ub = [y_ac.max()]  # Upper bound for prix_unitaire

    # Run PSO optimization
    best_params, _ = pso(objective_function, lb, ub, args=(X_ac, y_ac))

    optimized_params.append({'Code_AC': code_ac, 'optimized_prix_unitaire': best_params[0]})

# Merge the optimized prix_unitaire back to the original DataFrame
optimized_df = pd.DataFrame(optimized_params)
print(optimized_df)
# Create a bar plot to visualize the average 'optimized_prix_unitaire' for each unique 'Code_AC'
plt.figure(figsize=(12, 6))
sns.barplot(x='Code_AC', y='optimized_prix_unitaire', data=optimized_df)
plt.title('Optimized Prix Unitaire for each Code_AC')
plt.xlabel('Code_AC')
plt.ylabel('Optimized Prix Unitaire')
plt.show()



# Visualize the results
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Code_AC', y='optimized_prix_unitaire', data=optimized_df, label='Optimized Prix Unitaire', marker='X')
plt.title('Optimized Prix Unitaire for each Code_AC')
plt.xlabel('Activite')
plt.ylabel('Prix Unitaire')
plt.legend()
plt.show()
# Visualize the results
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Code_AC', y='prix_unitaire', data=df, label='Original Data')
sns.scatterplot(x='Code_AC', y='optimized_prix_unitaire', data=df, label='Optimized Prix Unitaire', marker='X')
plt.title('Original vs Optimized Prix Unitaire for each Code_AC')
plt.xlabel('Activite')
plt.ylabel('Prix Unitaire')
plt.legend()
plt.show()



'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Price_OS1.csv')
# Assuming your dataset is stored in a DataFrame named df
# Load your dataset and convert relevant columns to numeric

# Replace commas with dots and convert to numeric
df['prix_unitaire'] = pd.to_numeric(df['prix_unitaire'].str.replace(',', '.'), errors='coerce')
df['Code_AC'] = pd.to_numeric(df['Code_AC'], errors='coerce')  # Convert 'Code_AC' to numeric

# Visualize the relationship between 'Code_AC' and 'prix_unitaire' using a scatter plot
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x='Code_AC', y='prix_unitaire', data=df)
plt.title('Scatter Plot of Code_AC vs prix_unitaire')
plt.xlabel('Code_AC')
plt.ylabel('Prix Unitaire')

# Create a bar plot to visualize the average 'prix_unitaire' for each unique 'Code_AC'
plt.subplot(1, 2, 2)
average_prices = df.groupby('Code_AC')['prix_unitaire'].mean().reset_index()
sns.barplot(x='Code_AC', y='prix_unitaire', data=average_prices)
plt.title('Average Prix Unitaire for each Code_AC')
plt.xlabel('Code_AC')
plt.ylabel('Average Prix Unitaire')

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X = df[['Code_AC']]
y = df['prix_unitaire']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Code_AC', y='prix_unitaire', data=df, label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Code_AC vs prix_unitaire')
plt.xlabel('Code_AC')
plt.ylabel('Prix Unitaire')
plt.legend()
plt.show()
'''