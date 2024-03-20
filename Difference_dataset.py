'''
import pandas as pd
from sqlalchemy import create_engine

# SQL Server connection string
sql_server_connection_string = 'mssql+pyodbc://sa:oartcos@srv-cg/cosider_oa?driver=ODBC+Driver+17+for+SQL+Server'

# Replace the above connection string with your actual SQL Server connection details

# Read data from SQL Server
sql_query = 'SELECT * FROM Materiel.type'
sql_engine = create_engine(sql_server_connection_string)
sql_data = pd.read_sql(sql_query, sql_engine)

# Read data from Excel
excel_file_path = 'TYPE.xlsx'
excel_data = pd.read_excel(excel_file_path)

# Merge the two DataFrames on a common column (assuming 'ID' as the common column, replace it with your actual common column)
merged_data = pd.merge(sql_data, excel_data, on='TYPE', how='outer', indicator=True)

# Filter rows that are in the SQL Server table but not in the Excel file
sql_only_data = merged_data[merged_data['_merge'] == 'left_only'].drop(columns=['_merge'])

# Print the data
print("Data in SQL Server table but not in Excel file:")
print(sql_only_data)

# Save the result as an Excel file
output_excel_path = 'type_result.xlsx'
sql_only_data.to_excel(output_excel_path, index=False)
print(f"Result saved to {output_excel_path}")
'''



import pandas as pd

# Read data from Excel
sql_data = pd.read_excel('TYPE_oa.xlsx')
excel_data = pd.read_excel('TYPE.xlsx')

# Extract 'type' column from Excel data
#excel_type_column = excel_data['TYPE']

# Extract 'type' column from SQL data
#sql_type_column = sql_data['TYPE']


# Strip whitespace and convert to lowercase
# #Extract 'TYPE', 'LIBELLE', and 'Marque' columns from Excel data
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