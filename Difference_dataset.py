



import pandas as pd

# Read data from Excel
sql_data = pd.read_excel('TYPE_oa.xlsx')
excel_data = pd.read_excel('TYPE.xlsx')

# Extract 'type' column from Excel data
#excel_type_column = excel_data['TYPE']

# Extract 'type' column from SQL data
#sql_type_column = sql_data['TYPE']


# Strip whitespace and convert to lowercase
# Extract 'TYPE', 'LIBELLE', and 'Marque' columns from Excel data
#excel_data[['TYPE', 'LIBELLE', 'Marque']] = excel_data[['TYPE', 'LIBELLE', 'Marque']].apply(lambda x: x.str.strip().str.lower())
excel_data = excel_data['TYPE'].str.strip().str.lower()
sql_data = sql_data['TYPE'].str.strip().str.lower()

# Compare 'type' columns
differences = sql_data[~sql_data.isin(excel_data)]
print("Differences between 'type' columns:")
print(differences)
# Save the result as an Excel file
output_excel_path = 'type_result_all.xlsx'
differences.to_excel(output_excel_path, index=False)
'''
import pandas as pd

# Read data from Excel
sql_data = pd.read_excel('TYPE_oa.xlsx')
excel_data = pd.read_excel('TYPE.xlsx')

# Extract 'type' column from Excel data
excel_type_column = excel_data['TYPE']

# Extract 'type' column from SQL data
sql_type_column = sql_data['TYPE']


# Strip whitespace and convert to lowercase
excel_type_column = excel_data['TYPE'].str.strip().str.lower()
sql_type_column = sql_data['TYPE'].str.strip().str.lower()

# Find common values between 'type' columns
common_values = sql_type_column[sql_type_column.isin(excel_type_column)]

# Save common values to a new Excel file
output_excel_path = 'common_type_values.xlsx'
common_values.to_excel(output_excel_path, index=False)
print("Common 'type' values saved to:", output_excel_path)
'''