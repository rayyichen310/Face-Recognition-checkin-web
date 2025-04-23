import os
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import onnxruntime as ort
import io
import json
import logging

# --- Get the directory where utils.py is located ---
_script_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.normpath(os.path.join(_script_dir, '..'))
# --- End ---

# --- Settings Configuration ---
SETTINGS_FILE = os.path.join(_base_dir, 'config', 'settings.json')
DEFAULT_SETTINGS = {
    "db_subdir": "database",
    "db_filename": "face_embeddings.db",
    "image_source_dirname": "dataset",
    "recognition_threshold": 0.6,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "onnx_model_subdir": "model",
    "onnx_model_filename": "arcface_r100_v1.onnx",
    "blur_threshold": 100,
    "ratio_diff_threshold": 0.25
}

# --- 設定 Logger ---
logger = logging.getLogger(__name__)
# 建議在 app.py 中統一配置 logging
# logging.basicConfig(level=logging.INFO)
# --- 結束 Logger 設定 ---


def load_settings():
    """從 JSON 檔案載入設定，如果檔案不存在或無效則使用預設值。"""
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    if not os.path.exists(SETTINGS_FILE):
        logger.info(f"Settings file not found at {SETTINGS_FILE}. Creating with default settings.")
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            loaded_settings = json.load(f)
            settings = DEFAULT_SETTINGS.copy()
            settings.update(loaded_settings)
            for key, value in DEFAULT_SETTINGS.items():
                settings.setdefault(key, value) # 確保所有預設鍵存在
            logger.info(f"Settings loaded successfully from {SETTINGS_FILE}")
            return settings
    except (json.JSONDecodeError, IOError, TypeError) as e:
        logger.error(f"Failed to load settings from {SETTINGS_FILE}: {e}. Using default settings.", exc_info=True)
        try:
            save_settings(DEFAULT_SETTINGS) # 嘗試保存預設值
        except Exception as save_e:
            logger.error(f"Failed to save default settings after loading error: {save_e}", exc_info=True)
        return DEFAULT_SETTINGS.copy()


def save_settings(settings_to_save):
    """將設定儲存到 JSON 檔案。"""
    try:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_to_save, f, indent=4, ensure_ascii=False)
        logger.info(f"Settings saved successfully to {SETTINGS_FILE}")
        return True
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save settings to {SETTINGS_FILE}: {e}", exc_info=True)
        return False

# --- Load settings at startup ---
settings = load_settings()
# --- End Settings Configuration ---

# --- Update Setting Function ---
def update_setting(key, value):
    """
    更新單個設定項並保存。
    返回 True 表示成功，False 表示失敗。
    """
    global settings
    if key in settings:
        original_value = settings[key]
        default_type = type(DEFAULT_SETTINGS.get(key, original_value)) # 使用預設值或當前值的類型

        try:
            # 類型轉換邏輯
            if default_type is bool:
                if isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes', 'on']: value = True
                    elif value.lower() in ['false', '0', 'no', 'off']: value = False
                    else: raise ValueError(f"Cannot convert string '{value}' to boolean")
                else: value = bool(value)
            else:
                value = default_type(value)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert value '{value}' to type {default_type} for key '{key}': {e}")
            return False

        settings[key] = value
        if save_settings(settings):
            logger.info(f"Setting '{key}' updated to '{value}' and saved.")
            if key in ["min_detection_confidence", "min_tracking_confidence", "onnx_model_subdir", "onnx_model_filename", "db_subdir", "db_filename"]:
                 logger.warning(f"Changing '{key}' requires an application restart or manual re-initialization to take full effect.")
            return True
        else:
            settings[key] = original_value # Revert on save failure
            logger.error(f"Failed to save settings after attempting to update '{key}'. Value reverted.")
            return False
    else:
        logger.warning(f"Attempted to update non-existent setting key: '{key}'")
        return False
# --- End Update Setting Function ---


# --- NumPy Array Adapters for SQLite ---
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    # 增加對空字節串的處理
    if not text:
        return None
    try:
        return np.load(out, allow_pickle=True)
    except Exception as e:
        logger.error(f"Error in np.load within convert_array: {e}", exc_info=True)
        return None


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)
# --- End Adapters ---

# --- Landmark Indices ---
LANDMARK_INDICES_5 = [468, 473, 1, 61, 291] # L-eye, R-eye, Nose, L-mouth, R-mouth

# --- Initialize MediaPipe Face Mesh ---
detector = None
try:
    min_det_conf = settings.get("min_detection_confidence", DEFAULT_SETTINGS["min_detection_confidence"])
    min_track_conf = settings.get("min_tracking_confidence", DEFAULT_SETTINGS["min_tracking_confidence"])
    detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf
    )
    logger.info(f"MediaPipe FaceMesh initialized (min_detection_confidence: {min_det_conf}, min_tracking_confidence: {min_track_conf})")
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe FaceMesh: {e}", exc_info=True)
    # 如果偵測器至關重要，考慮引發異常或退出
# --- 結束初始化 ---

# --- ONNX 模型初始化 ---
onnx_model_subdir = settings.get("onnx_model_subdir", DEFAULT_SETTINGS["onnx_model_subdir"])
onnx_model_filename = settings.get("onnx_model_filename", DEFAULT_SETTINGS["onnx_model_filename"])
onnx_path = os.path.join(_base_dir, onnx_model_subdir, onnx_model_filename)
sess = None
try:
    if not os.path.exists(onnx_path):
         logger.warning(f"找不到 ONNX 模型檔案：{onnx_path}。辨識功能已停用。")
    else:
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        logger.info(f"ONNX 模型已載入：{onnx_path}")
except Exception as e:
    logger.error(f"從 {onnx_path} 載入 ONNX 模型時發生錯誤：{e}。辨識功能已停用。", exc_info=True)
    sess = None
# --- 結束 ONNX 模型初始化 ---

# --- 資料庫路徑 ---
db_subdir = settings.get("db_subdir", DEFAULT_SETTINGS["db_subdir"])
db_filename = settings.get("db_filename", DEFAULT_SETTINGS["db_filename"])
db_dir = os.path.join(_base_dir, db_subdir)
db_path = os.path.join(db_dir, db_filename)
# logger.debug(f"Database path set to: {db_path}") # 已移除除錯日誌
# --- 結束資料庫路徑 ---

# --- 圖片來源目錄 ---
image_source_dirname = settings.get("image_source_dirname", DEFAULT_SETTINGS["image_source_dirname"])
image_source_dir = os.path.join(_base_dir, image_source_dirname)
# logger.debug(f"Image source directory set to: {image_source_dir}") # 已移除除錯日誌
# --- 結束圖片來源目錄 ---

# 已移除 os.makedirs(db_dir) 的 try...except

logger.info("Utils 模組已初始化。")