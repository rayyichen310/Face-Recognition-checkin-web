import numpy as np
import sqlite3
from scipy.spatial.distance import cosine
import io
import logging
from typing import Tuple, Dict, Any, Optional

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NumPy 陣列的 SQLite 轉換器 ---
def convert_array(text: bytes) -> Optional[np.ndarray]:
    """將 SQLite BLOB 轉換回 NumPy 陣列。"""
    if not isinstance(text, bytes):
        logging.warning(f"convert_array received non-bytes input: {type(text)}")
        # 如果已經是 ndarray，直接返回；否則記錄錯誤並返回 None
        if isinstance(text, np.ndarray):
            return text
        else:
            logging.error("convert_array cannot handle non-bytes, non-ndarray input.")
            return None

    if not text: # 檢查是否為空字節
        logging.warning("convert_array received empty bytes.")
        return None

    out = io.BytesIO(text)
    out.seek(0)
    try:
        # allow_pickle=True 為了兼容性，但需注意安全風險
        return np.load(out, allow_pickle=True)
    except Exception as e:
        # 記錄轉換錯誤，但不在穩定運行時打印字節內容
        logging.error(f"Error in np.load within convert_array: {e}", exc_info=True)
        return None # 返回 None 表示轉換失敗

# --- 註冊 SQLite 轉換器 ---
# 雖然我們在 compare_face 中顯式調用 convert_array，但保留註冊可能對其他地方有用
# 或者如果確定只在此處使用，可以移除
sqlite3.register_converter("ARRAY", convert_array)

# --- 人臉比對函數 ---
def compare_face(embedding: np.ndarray, threshold: float, db_path: str) -> Tuple[str, float, Dict[str, float]]:
    """
    將輸入的人臉嵌入與資料庫中的嵌入進行比較。
    :param embedding: 要比較的單個人臉嵌入 (NumPy array)。
    :param threshold: 餘弦距離閾值。
    :param db_path: 資料庫檔案的路徑。
    :return: Tuple (matched_name, min_distance, all_results)
             - matched_name: 匹配的姓名 ('Unknown' 或 'Error: ...')
             - min_distance: 最小餘弦距離 (float('inf') 如果無有效比較)
             - all_results: 包含所有姓名及其對應距離的字典
    """
    db_data = []
    try:
        # 使用 'with' 陳述式確保連接自動關閉
        # detect_types 允許自動處理已註冊的轉換器 (雖然我們也手動調用)
        with sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn_db:
            cursor_db = conn_db.cursor()
            # 顯式請求轉換 embedding 欄位
            cursor_db.execute("SELECT name, embedding FROM face_embeddings")
            db_data = cursor_db.fetchall()
    except sqlite3.Error as db_err:
        logging.error(f"Failed to connect or fetch from database '{db_path}': {db_err}", exc_info=True)
        return "Error: DB Access", float('inf'), {}

    if not db_data:
        logging.warning("No embeddings found in the database.")
        return 'Unknown', float('inf'), {} # 資料庫為空

    try:
        # 確保輸入 embedding 是正確的類型和形狀
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 0 or embedding.size == 0:
             logging.error("Input embedding is invalid (empty or scalar).")
             return "Error: Invalid Input Embedding", float('inf'), {}

        total_distances = []
        total_names = []
        all_results = {}

        for name, db_embedding_blob in db_data:
            if db_embedding_blob is None:
                logging.warning(f"Database embedding for '{name}' is None. Skipping.")
                continue

            # 顯式調用轉換器
            db_embedding_np = convert_array(db_embedding_blob)

            if db_embedding_np is None:
                logging.warning(f"Failed to convert database embedding for '{name}'. Skipping.")
                continue

            try:
                # 確保資料庫讀取的 embedding 也是正確類型和形狀
                db_embedding_np = np.asarray(db_embedding_np, dtype=np.float32)
                if db_embedding_np.ndim == 0 or db_embedding_np.size == 0:
                    logging.warning(f"Converted database embedding for '{name}' is invalid. Skipping.")
                    continue

                # 計算餘弦距離
                distance = cosine(embedding, db_embedding_np)

                # 處理 NaN (可能由零向量引起)
                current_distance = distance if not np.isnan(distance) else float('inf')

                total_names.append(name)
                total_distances.append(current_distance)
                all_results[name] = current_distance

            except ValueError as ve:
                # 通常由形狀不匹配引起
                logging.warning(f"ValueError calculating distance for '{name}': {ve}. Skipping.")
                # 不將此無效比較添加到結果中
            except Exception as calc_err:
                # 捕捉其他可能的計算錯誤
                logging.error(f"Unexpected error calculating distance for '{name}': {calc_err}", exc_info=True)
                # 不將此無效比較添加到結果中


        if not total_distances:
            logging.warning("No valid distances were calculated.")
            return 'Unknown', float('inf'), all_results # 返回空的 all_results

        # 找到最小距離及其對應的名稱
        min_distance = min(total_distances)
        min_name_index = total_distances.index(min_distance)
        min_name = total_names[min_name_index]

        # 根據閾值確定最終匹配名稱
        matched_name = min_name if min_distance <= threshold else 'Unknown'

        return matched_name, min_distance, all_results

    except Exception as comp_err:
        # 捕捉在比較循環之外或 setup 階段發生的意外錯誤
        logging.error(f"Unexpected error during face comparison process: {comp_err}", exc_info=True)
        return f"Error: Comparison Failed", float('inf'), {}

