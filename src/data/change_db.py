"""
    メッシュからのデータ取得し、dbファイルにまとめたときにOPR列が途中で途切れて、違う列に格納されていたので、それを一つの列に統合知るためのコード
    """
import sqlite3

def update_weather_data():
    """
    Connects to the weather database and updates a specific column
    based on data from another column for a specified range of rows.
    """
    # --- Configuration ---
    # These are the only values you need to change for future updates.
    db_filename = 'weather_database.db'
    table_name = 'weather_data'
    source_column = 'OPRori'  # The column to copy data FROM
    target_column = 'OPR'                  # The column to copy data TO
    start_rowid = 104673                   # The starting row for the update
    # --- End Configuration ---

    conn = None  # Initialize connection to None
    try:
        # 1. Connect to the SQLite database
        conn = sqlite3.connect(db_filename)
        cursor = conn.cursor()
        print(f"Successfully connected to '{db_filename}'.")

        # 2. Construct the SQL UPDATE query
        # Using double quotes handles column names with spaces or special characters
        query = f"""
        UPDATE "{table_name}"
        SET "{target_column}" = "{source_column}"
        WHERE rowid >= {start_rowid};
        """

        print("Executing update...")
        # 3. Execute the query
        cursor.execute(query)

        # 4. Get the number of rows that were changed
        rows_affected = cursor.rowcount

        # 5. Commit (save) the changes to the database file
        conn.commit()

        print("\n✅ Update complete.")
        print(f"{rows_affected} rows were successfully updated in the '{target_column}' column.")

    except sqlite3.Error as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check if the database file, table name, and column names are correct.")

    finally:
        # 6. Ensure the database connection is always closed
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == '__main__':
    # This makes the script runnable from the command line
    update_weather_data()