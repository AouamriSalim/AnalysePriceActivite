import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("TCR_Non_Traite.csv")

print(data.head())

# drop rows with missing values
df_no_missing = data.dropna()

# OR
'''
## fill missing values with 0
df_filled = data.fillna(0)

#  OR
# interpolate missing values
df_interpolated = data.interpolate()

# OR
from sklearn.impute import SimpleImputer

# create a SimpleImputer object with mean imputation
imputer = SimpleImputer(strategy='mean')

# fit the imputer to the data and transform the data
df_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)  
'''
# remove duplicates based on all columns
data.drop_duplicates(inplace=True)

'''
# change column B to integer type

df['B'] = df['B'].astype(int)
'''
'''
 # Create an instance of LabelEncoder
le = LabelEncoder()

# Fit and transform the categorical data
data['Pole_Num'] = le.fit_transform(data['Pole'])
data['A_TCR_Num'] = le.fit_transform(data['Agrégat_TCR'])
'''
