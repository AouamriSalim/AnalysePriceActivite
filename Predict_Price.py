import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Assuming 'data_for_estimation' is your DataFrame
# Replace this with your actual DataFrame
file_path = 'D:\Work Cos\ReportPrixUnitMois.csv'
data_for_estimation = pd.read_csv(file_path)
# Convert categorical columns to numeric using Label Encoding
label_encoder = LabelEncoder()
data_for_estimation['Code_AC1_encoded'] = label_encoder.fit_transform(data_for_estimation['Code_AC1'])

# Drop unnecessary columns
X = data_for_estimation[['Code_AC1_encoded']]
y = data_for_estimation['prix_unitaire'].replace(',', '.', regex=True).astype(float)

# Handle NaN values in the target variable 'prix_unitaire'
imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Make predictions for the entire dataset
data_for_estimation['predicted_prix_unitaire'] = regression_model.predict(X)

# Print the predicted prix_unitaire for each Code_AC1 group
predicted_values = data_for_estimation.groupby('Code_AC1')['predicted_prix_unitaire'].mean().reset_index()
print(data_for_estimation)

