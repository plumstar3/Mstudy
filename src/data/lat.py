"""
FieldData.dbからplace,year,lat,lonを取得するためのプログラム。逆ジオコーティング用のエクセルファイルを作成するため。
"""
import pandas as pd
import sqlite3
import os

def extract_complete_locations_to_excel():
    """
    Connects to FieldData.db, extracts rows where place, year, lat, and lon
    are all non-NULL, and saves these columns to an Excel file.
    """
    
    # --- Configuration ---
    # Database path (assuming DB is in the same directory as the script)
    db_filename = 'FieldData.db'
    table_name = 'Questionaire'
    columns_to_extract = ['place', 'year', 'lat', 'lon'] # Columns needed
    output_excel_filename = 'locations_complete.xlsx'
    # ---------------------
    
    conn = None # Initialize connection
    try:
        # 1. Connect to the database
        conn = sqlite3.connect(db_filename)
        print(f"Successfully connected to '{db_filename}'.")
        
        # 2. Build the SQL query
        # Select the desired columns
        # Filter rows where ANY of the specified columns are NULL
        query = f"""
        SELECT {', '.join(columns_to_extract)}
        FROM "{table_name}"
        WHERE place IS NOT NULL
          AND year IS NOT NULL
          AND lat IS NOT NULL
          AND lon IS NOT NULL
          AND yield IS NOT NULL
          AND seed_date IS NOT NULL
          AND harvest_date IS NOT NULL;
        """
        
        # 3. Read data into a pandas DataFrame
        print("Executing query to extract complete location data...")
        locations_df = pd.read_sql_query(query, conn)
        
        print(f"Found {len(locations_df)} rows with complete data.")
        
        # 4. Save the DataFrame to an Excel file
        if not locations_df.empty:
            print(f"Saving data to '{output_excel_filename}'...")
            # Use encoding='utf-8-sig' for better compatibility with Excel
            locations_df.to_excel(output_excel_filename, index=False)
            print("✅ Successfully saved to Excel.")
        else:
            print("No complete data found to save.")
            
    except sqlite3.Error as e:
        print(f"\n❌ Database error: {e}")
        print("Please check the database file path and table/column names.")
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        
    finally:
        # 5. Close the database connection
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    extract_complete_locations_to_excel()