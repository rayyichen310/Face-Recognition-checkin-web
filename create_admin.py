import sqlite3
import getpass
import sys
import os
from werkzeug.security import generate_password_hash

# --- 設定 src 目錄的路徑 ---
# 假設此腳本位於專案根目錄，src 在其下一層
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# --- 結束路徑設定 ---

try:
    # 從您的 database 模組匯入必要的函數和資料庫路徑
    from database import create_connection, add_user, get_user_by_username
    from utils import db_path # 從 utils 取得資料庫路徑
except ImportError as e:
    print(f"Error: Could not import necessary modules. Ensure this script is in the project root and the 'src' directory structure is correct.")
    print(f"Details: {e}")
    sys.exit(1)

def create_admin_user():
    """Prompt the user for information and create an admin account."""
    print("--- Create Admin User ---")
    username = input("Enter admin username: ")
    password = getpass.getpass("Enter admin password: ")
    password_confirm = getpass.getpass("Confirm password: ")

    if not username or not password:
        print("Error: Username and password cannot be empty.")
        return

    if password != password_confirm:
        print("Error: Passwords do not match.")
        return

    # Hash the password
    password_hash = generate_password_hash(password)

    conn = None
    try:
        conn = create_connection(db_path)
        if conn is None:
            print(f"Error: Could not connect to database {db_path}")
            return

        cursor = conn.cursor()

        # Check if the user already exists
        existing_user = get_user_by_username(conn, username)
        if existing_user:
            print(f"User '{username}' already exists. Do you want to set this user as admin? (y/n): ", end='')
            choice = input().lower()
            if choice == 'y':
                # Update existing user to admin
                try:
                    cursor.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (username,))
                    conn.commit()
                    print(f"User '{username}' successfully set as admin.")
                except sqlite3.Error as e:
                    print(f"Error updating user '{username}' to admin: {e}")
            else:
                print("Operation cancelled.")
            return

        # If the user does not exist, add the user
        user_id = add_user(conn, username, password_hash)
        if user_id:
            print(f"User '{username}' created successfully (ID: {user_id}).")
            # Set the newly created user as admin
            try:
                cursor.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))
                conn.commit()
                print(f"User '{username}' successfully set as admin.")
            except sqlite3.Error as e:
                print(f"Error setting new user '{username}' as admin: {e}")
        else:
            # add_user likely handled the error message internally
            pass

    except sqlite3.Error as e:
        print(f"Database error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Ensure the database and tables exist (if not automatically created in database.py)
    # You may need to add logic in database.py to ensure tables exist, or call create_table here
    # Example:
    # conn = create_connection(db_path)
    # if conn:
    #     create_table(conn, USERS_TABLE_SQL) # Assuming USERS_TABLE_SQL is imported
    #     conn.close()

    create_admin_user()