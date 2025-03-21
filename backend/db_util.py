#!/usr/bin/env python3
"""
Database utility script for viewing and managing the SQLite database.
Run this script with Python to view all records in the database.
"""

import sqlite3
import sys
from tabulate import tabulate

def print_database(db_path="geospatial.db"):
    """Print all records in the database in a tabular format"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"\n=== TABLE: {table_name} ===")
            
            # Get column names
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Get all rows
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()
            
            if rows:
                print(tabulate(rows, headers=columns, tablefmt="grid"))
            else:
                print("No records found.")
        
        conn.close()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def clear_database(db_path="geospatial.db"):
    """Clear all records from the database while preserving the schema"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            if table_name != "sqlite_sequence":  # Skip SQLite's internal table
                cursor.execute(f"DELETE FROM {table_name};")
        
        conn.commit()
        conn.close()
        print("Database cleared successfully.")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        clear_database()
    else:
        print_database() 