import pyodbc
from sqlalchemy import create_engine
import pandas as pd

# Database connection parameters
server = 'SRV'
database = 'staging'
username = 'user'
password = 'password'
driver = 'ODBC Driver 17 for SQL Server'  # Change this based on your SQL Server version

# Specify the target table and column in the database
target_table = 'Parc'  # Replace with your actual target table name
target_schema = 'schema'  # Replace with your actual target schema name
target_column = 'target_column'  # Replace with your actual target column name

# Create a SQLAlchemy engine
engine = create_engine(f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}")

try:
    # SQL query to select values
    select_query = """
        SELECT
            LEFT(Matricule, 3) AS Pole,
            CODE_groupe, Mmaa,
            COALESCE(SUM(montant) / 1000, 0) AS variable
        FROM
            RH.Tab_Archive_paie_partielle
        WHERE
            Code_Rubrique IN ('341', '675') AND LEFT(Mmaa, 4) = '2023' AND LEFT(Matricule, 3) = 'F00'
        GROUP BY
            LEFT(Matricule, 3),
            CODE_groupe, Mmaa
    """

    # Execute the SQL query and load the results into a pandas DataFrame
    df = pd.read_sql(select_query, con=engine)

    print(f"DataFrame loaded with values from '{target_schema}.{target_table}'.")

except Exception as e:
    print(f"Error in SELECT query: {e}")

finally:
    # Dispose of the SQLAlchemy engine in the 'finally' block
    engine.dispose()

try:
    # Update values in the target column using SQL UPDATE statement
    update_query = f"""
        UPDATE {target_schema}.{target_table}
        SET {target_column} = src.variable
        FROM (
            {select_query}
        ) AS src
        WHERE
            {target_schema}.{target_table}.Pole = src.Pole
            AND {target_schema}.{target_table}.CODE_groupe = src.CODE_groupe
            AND {target_schema}.{target_table}.Mmaa = src.Mmaa
    """

    # Execute the SQL query to update values
    with engine.connect() as connection:
        connection.execute(update_query)

    print(f"Values in column '{target_column}' updated in the table '{target_schema}.{target_table}'.")

except Exception as e:
    print(f"Error in UPDATE query: {e}")

finally:
    # Dispose of the SQLAlchemy engine in the 'finally' block
    engine.dispose()
