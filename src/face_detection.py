import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any  # Import Any for detector type hint

# --- Logging Configuration ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Module Imports (Ensure utils is accessible) ---
try:
    from utils import LANDMARK_INDICES_5
except ImportError:
    logging.error("Failed to import LANDMARK_INDICES_5 from utils. Ensure utils.py is accessible.")
    raise  # Re-raise, as this is critical

# --- Helper Functions ---
def _calculate_bbox(landmarks_all: List[Any], img_shape: Tuple[int, int, int]) -> List[int]:
    """根據所有 MediaPipe 關鍵點計算邊界框。"""
    h, w, _ = img_shape
    # 假設 landmarks_all 中的 lm 物件總是有效
    x_coords = [lm.x * w for lm in landmarks_all]
    y_coords = [lm.y * h for lm in landmarks_all]
    if not x_coords or not y_coords:  # 處理空列表
        return [0, 0, w, h]  # 返回整個圖像作為備用
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # 添加邊距
    padding = 0.05  # 5% 邊距
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - width * padding)
    y_min = max(0, y_min - height * padding)
    x_max = min(w, x_max + width * padding)
    y_max = min(h, y_max + height * padding)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def _extract_5_landmarks(landmarks_all: List[Any], img_shape: Tuple[int, int, int]) -> Optional[List[List[int]]]:
    """使用 LANDMARK_INDICES_5 從所有 MediaPipe 關鍵點中提取 5 個關鍵點。"""
    h, w, _ = img_shape
    landmarks_5 = []
    max_required_index = max(LANDMARK_INDICES_5)

    # 檢查是否有足夠的關鍵點
    if len(landmarks_all) <= max_required_index:
        return None

    # 假設索引和屬性總是有效的
    for idx in LANDMARK_INDICES_5:
        lm = landmarks_all[idx]
        # 檢查座標是否在有效範圍內
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            return None  # 如果座標無效，則此人臉的關鍵點提取失敗
        landmarks_5.append([int(lm.x * w), int(lm.y * h)])  # 轉換為像素座標

    return landmarks_5

# --- Main Detection Functions ---
def face_detect(img_path: str, detector: Any) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    """
    使用 MediaPipe 從圖片檔案中偵測人臉和關鍵點。

    :param img_path: 圖片檔案的路徑。
    :param detector: 已初始化的 MediaPipe FaceMesh 實例。
    :return: Tuple (img_rgb, detections_list)
             img_rgb: RGB 格式的圖片 (NumPy 陣列) 或 None (如果讀取失敗)。
             detections_list: 包含偵測結果的列表，或空列表。
    """
    # 檔案存在性檢查
    if not os.path.exists(img_path):
        logging.error(f"File not found: {img_path}")
        return None, []

    # 讀取圖片 (保留 try-except)
    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception as read_err:
        logging.error(f"Error reading or decoding image file {img_path}: {read_err}", exc_info=True)
        return None, []

    if img_bgr is None:
        logging.warning(f"Failed to decode image: {img_path}. It might be corrupted or unsupported.")
        return None, []

    # 調用處理 BGR 圖像的函數
    return face_detect_bgr(img_bgr, detector)

def face_detect_bgr(img_bgr: np.ndarray, detector: Any) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
    """
    使用 MediaPipe 從 BGR 格式的圖片中偵測人臉和關鍵點。

    :param img_bgr: BGR 格式的圖片 (NumPy 陣列)。
    :param detector: 已初始化的 MediaPipe FaceMesh 實例。
    :return: Tuple (img_rgb, detections_list)
             img_rgb: RGB 格式的圖片 (NumPy 陣列) 或 None (如果轉換失敗)。
             detections_list: 包含偵測結果的列表，或空列表。
    """
    detections_list = []
    img_rgb = None
    try:  # 保留對核心操作的 try-except
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        # 使用 MediaPipe 處理圖片
        img_rgb.flags.writeable = False  # 提高效能
        results = detector.process(img_rgb)
        img_rgb.flags.writeable = True  # 恢復

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_all = face_landmarks.landmark

                # 計算邊界框
                bbox = _calculate_bbox(landmarks_all, img_rgb.shape)

                # 提取 5 個關鍵點
                landmarks_5 = _extract_5_landmarks(landmarks_all, img_rgb.shape)

                if landmarks_5:  # 僅當成功提取 5 個關鍵點時才添加
                    detection_info = {
                        'bbox': bbox,
                        'landmarks': landmarks_5
                    }
                    detections_list.append(detection_info)

        return img_rgb, detections_list

    except Exception as e:  # 捕捉其他所有可能的錯誤 (例如 MediaPipe)
        logging.error(f"Unexpected error during face detection processing: {e}", exc_info=True)
        # 如果 img_rgb 已創建，返回它，否則返回 None
        return img_rgb if img_rgb is not None else None, []