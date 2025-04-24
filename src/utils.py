import os
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import onnxruntime as ort
import io
import json
import logging

# --- 專案路徑設定 ---
_script_dir = os.path.dirname(os.path.abspath(__file__)) # utils.py 所在目錄
_base_dir = os.path.normpath(os.path.join(_script_dir, '..')) # 專案根目錄 (上一層)
# --- 結束 ---

# --- 應用程式設定 ---
SETTINGS_FILE = os.path.join(_base_dir, 'config', 'settings.json') # 設定檔完整路徑
DEFAULT_SETTINGS = { # 預設設定值
    "db_subdir": "database",                # 資料庫子目錄名稱
    "db_filename": "face_embeddings.db",    # 資料庫檔案名稱
    "image_source_dirname": "dataset",      # 圖片來源目錄名稱
    "recognition_threshold": 0.6,           # 臉部辨識相似度閾值
    "min_detection_confidence": 0.5,        # MediaPipe 臉部偵測最小信心度
    "min_tracking_confidence": 0.5,         # MediaPipe 臉部追蹤最小信心度
    "onnx_model_subdir": "model",           # ONNX 模型子目錄名稱
    "onnx_model_filename": "arcface_r100_v1.onnx", # ONNX 模型檔案名稱
    "blur_threshold": 100,                  # 模糊度檢測閾值 (值越小越嚴格)
    "ratio_diff_threshold": 0.25,           # 臉部關鍵點比例差異閾值
    "recognition_mode": "continuous"        # 即時辨識模式: 'continuous' (持續) 或 'lock_on' (鎖定)
}

# --- Logger 設定 ---
logger = logging.getLogger(__name__) # 取得此模組的 logger
# (建議在 app.py 中統一設定 logging 等級和格式)
# --- 結束 ---


def load_settings():
    """載入 JSON 設定檔。若失敗或檔案不存在，則使用預設值並嘗試儲存。"""
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True) # 確保設定檔目錄存在

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded_settings = json.load(f) # 嘗試載入 JSON

            # 合併設定：以預設值為基礎，用載入的值更新，確保所有預設鍵存在
            settings = DEFAULT_SETTINGS.copy()
            settings.update(loaded_settings)
            for key, default_value in DEFAULT_SETTINGS.items():
                settings.setdefault(key, default_value) # 若載入的設定缺少預設鍵，則補上

            logger.info(f"Successfully loaded settings from {SETTINGS_FILE}") # 成功從 {SETTINGS_FILE} 載入設定
            return settings # 返回成功載入並合併的設定

        except (json.JSONDecodeError, IOError, TypeError) as e:
            # 處理 JSON 格式錯誤、檔案讀取錯誤或類型錯誤
            logger.error(f"Failed to load settings from {SETTINGS_FILE}: {e}. Using default settings.", exc_info=True) # 從 {SETTINGS_FILE} 載入設定失敗: {e}。將使用預設設定。
            save_settings(DEFAULT_SETTINGS) # 嘗試用預設值覆蓋可能損壞的檔案
            # 繼續執行到函數末尾以返回預設值
    else:
        # 設定檔不存在
        logger.info(f"Settings file not found at {SETTINGS_FILE}. Creating with default settings.") # 在 {SETTINGS_FILE} 找不到設定檔。將使用預設設定建立。
        save_settings(DEFAULT_SETTINGS) # 嘗試建立預設設定檔
        # 繼續執行到函數末尾以返回預設值

    # 若檔案不存在或載入失敗，返回預設設定的副本
    return DEFAULT_SETTINGS.copy()


def save_settings(settings_to_save):
    """將設定字典儲存到 JSON 檔案。"""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True) # 確保目錄存在
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            # 以美觀的縮排格式寫入 JSON
            json.dump(settings_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Settings successfully saved to {SETTINGS_FILE}") # 設定成功儲存至 {SETTINGS_FILE}
        return True # 儲存成功
    except (IOError, TypeError) as e:
        # 處理檔案寫入錯誤或傳入類型錯誤
        logger.error(f"Failed to save settings to {SETTINGS_FILE}: {e}", exc_info=True) # 儲存設定至 {SETTINGS_FILE} 失敗: {e}
        return False # 儲存失敗

# --- 程式啟動時載入全域設定 ---
settings = load_settings()
# --- 結束 ---

# --- 更新單項設定 ---
def update_setting(key, value):
    """更新單個設定項，進行類型轉換並儲存。成功返回 True，失敗返回 False。"""
    global settings
    if key not in settings:
        logger.warning(f"Attempting to update non-existent setting key: '{key}'") # 嘗試更新不存在的設定鍵: '{key}'
        return False

    original_value = settings[key]
    # 取得目標類型 (優先使用預設值的類型，若無則用當前值的類型)
    default_type = type(DEFAULT_SETTINGS.get(key, original_value))

    try:
        # --- 類型轉換 ---
        if default_type is bool: # 特殊處理布林值 (接受多種字串表示)
            if isinstance(value, str):
                if value.lower() in ['true', '1', 'yes', 'on']: value = True
                elif value.lower() in ['false', '0', 'no', 'off']: value = False
                else: raise ValueError(f"Cannot convert string '{value}' to boolean") # 無法將字串 '{value}' 轉換為布林值
            else: value = bool(value) # 其他類型直接轉布林
        else:
            value = default_type(value) # 其他類型直接轉換
        # --- 轉換結束 ---
    except (ValueError, TypeError) as e:
        logger.error(f"Cannot convert value '{value}' to target type {default_type} for key '{key}': {e}") # 無法將值 '{value}' 轉換為鍵 '{key}' 的目標類型 {default_type}: {e}
        return False # 類型轉換失敗

    # 更新全域設定字典
    settings[key] = value
    if save_settings(settings): # 嘗試儲存更新後的設定
        logger.info(f"Setting '{key}' updated to '{value}' and saved.") # 設定 '{key}' 已更新為 '{value}' 並儲存。
        # 對需要重啟才能生效的設定發出警告
        if key in ["min_detection_confidence", "min_tracking_confidence", "onnx_model_subdir", "onnx_model_filename", "db_subdir", "db_filename"]:
             logger.warning(f"Changing '{key}' requires restarting the application or manual re-initialization to take full effect.") # 更改 '{key}' 需要重新啟動應用程式或手動重新初始化才能完全生效。
        return True # 更新並儲存成功
    else:
        settings[key] = original_value # 儲存失敗，還原更改
        logger.error(f"Failed to save settings after attempting to update '{key}'. Value reverted.") # 嘗試更新 '{key}' 後儲存設定失敗。值已還原。
        return False # 儲存失敗
# --- 結束 ---


# --- SQLite 的 NumPy 陣列轉換器 ---
def adapt_array(arr):
    """將 NumPy 陣列轉換為 SQLite 的 BLOB 類型。"""
    out = io.BytesIO()
    np.save(out, arr) # 將陣列儲存到 BytesIO 物件
    out.seek(0)
    return sqlite3.Binary(out.read()) # 讀取 BytesIO 並轉為 Binary

def convert_array(text):
    """將 SQLite 的 BLOB 類型轉換回 NumPy 陣列。"""
    if not text: # 處理空字節串的情況
        return None
    out = io.BytesIO(text)
    out.seek(0)
    try:
        # 從 BytesIO 物件載入 NumPy 陣列 (允許 pickle 是為了兼容舊格式，但有安全風險)
        return np.load(out, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error loading NumPy array in convert_array: {e}", exc_info=True) # 在 convert_array 中載入 NumPy 陣列時出錯: {e}
        return None # 載入失敗返回 None

# 註冊轉換器
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)
# --- 結束 ---

# --- MediaPipe 臉部關鍵點索引 (用於對齊) ---
LANDMARK_INDICES_5 = [468, 473, 1, 61, 291] # 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角

# --- 初始化 MediaPipe 臉部偵測器 ---
detector = None
try:
    # 從設定讀取信心度閾值
    min_det_conf = settings.get("min_detection_confidence", DEFAULT_SETTINGS["min_detection_confidence"])
    min_track_conf = settings.get("min_tracking_confidence", DEFAULT_SETTINGS["min_tracking_confidence"])
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,      # 使用影片模式
        max_num_faces=5,              # 最多偵測 5 張臉
        refine_landmarks=True,        # 細化眼睛和嘴唇周圍的關鍵點
        min_detection_confidence=min_det_conf, # 偵測信心度閾值
        min_tracking_confidence=min_track_conf   # 追蹤信心度閾值
    )
    logger.info(f"MediaPipe FaceMesh initialized (Detection Confidence: {min_det_conf}, Tracking Confidence: {min_track_conf})") # MediaPipe FaceMesh 已初始化 (偵測信心度: {min_det_conf}, 追蹤信心度: {min_track_conf})
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe FaceMesh: {e}", exc_info=True) # 初始化 MediaPipe FaceMesh 失敗: {e}
    # 若偵測器無法初始化，相關功能將無法使用
# --- 結束 ---

# --- 初始化 ONNX 臉部辨識模型 ---
onnx_model_subdir = settings.get("onnx_model_subdir", DEFAULT_SETTINGS["onnx_model_subdir"])
onnx_model_filename = settings.get("onnx_model_filename", DEFAULT_SETTINGS["onnx_model_filename"])
onnx_path = os.path.join(_base_dir, onnx_model_subdir, onnx_model_filename) # 模型完整路徑
sess = None # ONNX 推理會話
try:
    if not os.path.exists(onnx_path):
         logger.warning(f"ONNX model file not found: {onnx_path}. Face recognition feature will be disabled.") # 找不到 ONNX 模型檔案：{onnx_path}。臉部辨識功能將停用。
    else:
        # 使用 CPU 進行推理
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        logger.info(f"ONNX model loaded: {onnx_path}") # ONNX 模型已載入：{onnx_path}
except Exception as e:
    logger.error(f"Failed to load ONNX model from {onnx_path}: {e}. Face recognition feature will be disabled.", exc_info=True) # 從 {onnx_path} 載入 ONNX 模型失敗：{e}。臉部辨識功能將停用。
    sess = None # 確保載入失敗時 sess 為 None
# --- 結束 ---

# --- 資料庫路徑設定 ---
db_subdir = settings.get("db_subdir", DEFAULT_SETTINGS["db_subdir"])
db_filename = settings.get("db_filename", DEFAULT_SETTINGS["db_filename"])
db_dir = os.path.join(_base_dir, db_subdir) # 資料庫目錄
db_path = os.path.join(db_dir, db_filename) # 資料庫完整路徑
# --- 結束 ---

# --- 圖片來源目錄設定 ---
image_source_dirname = settings.get("image_source_dirname", DEFAULT_SETTINGS["image_source_dirname"])
image_source_dir = os.path.join(_base_dir, image_source_dirname) # 圖片來源目錄完整路徑
# --- 結束 ---

# (已移除不必要的資料庫目錄建立 try...except)

logger.info("Utils module initialized.") # Utils 模組初始化完成。