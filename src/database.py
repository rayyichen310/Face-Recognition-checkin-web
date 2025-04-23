import os
import sqlite3
import numpy as np
import io
import logging
from sqlite3 import Error
from typing import List, Dict, Any, Optional, Union, Tuple

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Module Imports ---
try:
    from utils import db_path
except ImportError:
    logging.error("Failed to import db_path from utils. Ensure utils.py is accessible.")
    raise

# --- Table Definitions ---
FACE_EMBEDDING_TABLE_SQL = """ CREATE TABLE IF NOT EXISTS face_embeddings (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL,
                                    embedding blob NOT NULL,
                                    source_image_path text
                                ); """

CHECK_IN_TABLE_SQL = """ CREATE TABLE IF NOT EXISTS check_ins (
                            id integer PRIMARY KEY AUTOINCREMENT,
                            user_name text NOT NULL,
                            timestamp text NOT NULL,
                            type text NOT NULL CHECK(type IN ('in', 'out'))
                        ); """

USERS_TABLE_SQL = """ CREATE TABLE IF NOT EXISTS users (
                        id integer PRIMARY KEY AUTOINCREMENT,
                        username text UNIQUE NOT NULL,
                        password_hash text NOT NULL,
                        is_admin integer NOT NULL DEFAULT 0
                    ); """

# --- NumPy Array SQLite Adapters/Converters ---
def adapt_array(arr: np.ndarray) -> sqlite3.Binary:
    """將 NumPy 陣列轉換為 SQLite BLOB。"""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text: bytes) -> Optional[np.ndarray]:
    """將 SQLite BLOB 轉換回 NumPy 陣列。"""
    if not text:
        logging.warning("Attempting to convert empty bytes to NumPy array.")
        return None
    out = io.BytesIO(text)
    out.seek(0)
    try:
        return np.load(out, allow_pickle=True)
    except Exception as e:
        logging.error(f"Error converting bytes to NumPy array: {e}", exc_info=True)
        logging.debug(f"Bytes prefix causing error: {text[:50]}")
        return None

# Register SQLite adapters/converters
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

# --- Database Utility Functions ---
def create_connection(db_file: str = db_path) -> Optional[sqlite3.Connection]:
    """建立到 SQLite 資料庫的連接。"""
    conn = None
    try:
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        logging.debug(f"Database connection established to {db_file}")
        return conn
    except Error as e:
        logging.error(f"Error connecting to database '{db_file}': {e}", exc_info=True)
        return None

def table_exists(db_file: str, table_name: str) -> bool:
    """檢查指定的 SQLite 資料庫中是否存在指定的表格。"""
    if not os.path.exists(db_file):
        return False
    conn = None
    try:
        conn = create_connection(db_file)
        if conn is None: return False
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        return cursor.fetchone() is not None
    except Error as e:
        logging.error(f"Error checking if table '{table_name}' exists in '{db_file}': {e}", exc_info=True)
        return False
    finally:
        if conn:
            conn.close()

def create_table(conn: sqlite3.Connection, create_table_sql: str) -> bool:
    """使用提供的 SQL 語句建立表格。"""
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        table_name = create_table_sql.split('IF NOT EXISTS')[-1].split('(')[0].strip()
        logging.info(f"Table '{table_name}' created or already exists.")
        return True
    except Error as e:
        logging.error(f"Error creating table using SQL: {create_table_sql[:50]}... : {e}", exc_info=True)
        return False

def initialize_database_tables(db_file: str = db_path):
    """確保所有必要的表格都存在於資料庫中。"""
    logging.info(f"Initializing database tables in {db_file}...")
    conn = create_connection(db_file)
    if conn:
        try:
            create_table(conn, FACE_EMBEDDING_TABLE_SQL)
            create_table(conn, CHECK_IN_TABLE_SQL)
            create_table(conn, USERS_TABLE_SQL)
        finally:
            conn.close()
            logging.info("Database table initialization check complete.")
    else:
        logging.error("Failed to initialize database tables due to connection error.")

# --- Data Manipulation Functions ---

# --- Face Embeddings ---
def add_face_embedding(conn: sqlite3.Connection, name: str, embedding: np.ndarray, source_image_path: Optional[str] = None) -> Optional[int]:
    """將人臉嵌入新增到資料庫。"""
    sql = ''' INSERT INTO face_embeddings(name, embedding, source_image_path)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    try:
        cur.execute(sql, (name, embedding, source_image_path))
        conn.commit()
        logging.info(f"Added face embedding for '{name}'. ID: {cur.lastrowid}")
        return cur.lastrowid
    except Error as e:
        logging.error(f"Error adding face embedding for '{name}': {e}", exc_info=True)
        conn.rollback()
        return None

def get_all_embeddings(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """查詢所有臉部嵌入，並確保嵌入是 NumPy 陣列。"""
    embeddings_list = []
    cur = conn.cursor()
    try:
        cur.execute("SELECT name, embedding as array, source_image_path FROM face_embeddings")
        rows = cur.fetchall()
        for row in rows:
            name, embedding_array, source_path = row
            if embedding_array is not None and isinstance(embedding_array, np.ndarray):
                embeddings_list.append({
                    "name": name,
                    "embedding": embedding_array,
                    "source_image_path": source_path
                })
            else:
                logging.warning(f"Failed to convert or invalid embedding for user '{name}' from source '{source_path}'. Skipping.")
    except Error as e:
        logging.error(f"Error fetching face embeddings: {e}", exc_info=True)
    return embeddings_list

# --- Check-ins ---
def add_check_in_record(conn: sqlite3.Connection, user_name: str, timestamp: str, check_in_type: str) -> Optional[int]:
    """將打卡紀錄新增到 check_ins 表格。"""
    sql = ''' INSERT INTO check_ins(user_name, timestamp, type)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    try:
        cur.execute(sql, (user_name, timestamp, check_in_type))
        conn.commit()
        logging.info(f"Added check-in record for '{user_name}' ({check_in_type}). ID: {cur.lastrowid}")
        return cur.lastrowid
    except Error as e:
        logging.error(f"Error adding check-in record for '{user_name}': {e}", exc_info=True)
        conn.rollback()
        return None

def get_all_check_ins(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """查詢所有打卡記錄，按時間降冪排序。"""
    records = []
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, user_name, timestamp, type FROM check_ins ORDER BY timestamp DESC")
        rows = cur.fetchall()
        records = [{"id": row[0], "name": row[1], "timestamp": row[2], "type": row[3]} for row in rows]
    except Error as e:
        logging.error(f"Error fetching all check-ins: {e}", exc_info=True)
    return records

def get_check_ins_by_username(conn: sqlite3.Connection, username: str) -> List[Dict[str, Any]]:
    """查詢特定姓名的打卡記錄，按時間降冪排序。"""
    records = []
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, user_name, timestamp, type FROM check_ins WHERE user_name=? ORDER BY timestamp DESC", (username,))
        rows = cur.fetchall()
        records = [{"id": row[0], "name": row[1], "timestamp": row[2], "type": row[3]} for row in rows]
    except Error as e:
        logging.error(f"Error fetching check-ins for user '{username}': {e}", exc_info=True)
    return records

# --- Users ---
def add_user(conn: sqlite3.Connection, username: str, password_hash: str, is_admin: bool = False) -> Optional[int]:
    """新增使用者到 users 表格。"""
    sql = ''' INSERT INTO users(username, password_hash, is_admin)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    try:
        cur.execute(sql, (username, password_hash, 1 if is_admin else 0))
        conn.commit()
        logging.info(f"User '{username}' added successfully. ID: {cur.lastrowid}")
        return cur.lastrowid
    except sqlite3.IntegrityError:
        logging.warning(f"Attempted to add existing username '{username}'.")
        conn.rollback()
        return None
    except Error as e:
        logging.error(f"Error adding user '{username}': {e}", exc_info=True)
        conn.rollback()
        return None

def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[Dict[str, Any]]:
    """根據使用者名稱查詢使用者。"""
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, username, password_hash, is_admin FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        if row:
            user_data = {
                "id": row[0],
                "username": row[1],
                "password_hash": row[2],
                "is_admin": bool(row[3])
            }
            return user_data
        else:
            return None
    except Error as e:
        logging.error(f"Error fetching user '{username}': {e}", exc_info=True)
        return None

def get_all_users(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """查詢所有已註冊的使用者。"""
    users = []
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, username, is_admin FROM users ORDER BY username")
        rows = cur.fetchall()
        users = [{"id": row[0], "username": row[1], "is_admin": bool(row[2])} for row in rows]
    except Error as e:
        logging.error(f"Error fetching all users: {e}", exc_info=True)
    return users

def delete_user(conn: sqlite3.Connection, username: str) -> bool:
    """刪除使用者帳號以及關聯的臉部特徵 (在一個事務中)。"""
    cur = conn.cursor()
    try:
        cur.execute("BEGIN TRANSACTION;")
        sql_delete_embeddings = "DELETE FROM face_embeddings WHERE name = ?"
        cur.execute(sql_delete_embeddings, (username,))
        logging.info(f"Deleted face embeddings associated with user '{username}'. Count: {cur.rowcount}")
        cur.execute("DELETE FROM users WHERE username=?", (username,))
        user_deleted_count = cur.rowcount
        conn.commit()
        if user_deleted_count > 0:
            logging.info(f"Successfully deleted user account '{username}'.")
            return True
        else:
            logging.warning(f"Attempted to delete non-existent user '{username}'.")
            return False
    except Error as e:
        logging.error(f"Error deleting user '{username}': {e}", exc_info=True)
        try:
            conn.rollback()
            logging.info(f"Transaction rolled back for user deletion '{username}'.")
        except Error as rb_err:
            logging.error(f"Error during rollback for user deletion '{username}': {rb_err}")
        return False

# --- 清空表格 ---
def clear_table(conn: sqlite3.Connection, table_name: str) -> bool:
    """清空指定表格的內容。"""
    allowed_tables_to_clear = {'check_ins'}
    if table_name not in allowed_tables_to_clear:
        logging.error(f"Attempted to clear disallowed table '{table_name}'. Operation aborted.")
        return False

    cur = conn.cursor()
    try:
        cur.execute(f"DELETE FROM {table_name}")
        conn.commit()
        logging.warning(f"Table '{table_name}' cleared successfully. Rows affected: {cur.rowcount}")
        return True
    except Error as e:
        logging.error(f"Error clearing table '{table_name}': {e}", exc_info=True)
        conn.rollback()
        return False

# --- Main execution block removed ---

