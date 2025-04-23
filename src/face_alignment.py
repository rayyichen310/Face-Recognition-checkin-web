import cv2
import numpy as np
import os


# --- Debugging Flag ---
SAVE_ALIGNED_WITH_TRANSFORMED_DST_POINTS = True
DEBUG_ALIGN_OUTPUT_DIR = "debug_alignment_output"
if SAVE_ALIGNED_WITH_TRANSFORMED_DST_POINTS:
    os.makedirs(DEBUG_ALIGN_OUTPUT_DIR, exist_ok=True)

# --- 臉部對齊函數 ---
def face_align(img_rgb: np.ndarray, landmarks: list) -> np.ndarray:
    """
    使用提供的 5 個關鍵點進行臉部對齊。
    :param img_rgb: 輸入的 RGB 圖像 (NumPy array)。
    :param landmarks: 5 個關鍵點 [[x1,y1], ..., [x5,y5]]。
    :return: 對齊後的 112x112 RGB 圖像 (NumPy array)。
    """
    # 目標 (dst) 關鍵點 (基於 112x112 模板)
    dst = np.array([
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041]   # 右嘴角
    ], dtype=np.float32)

    # 來源 (src) 關鍵點
    src = np.array(landmarks, dtype=np.float32).reshape(5, 2)

    # 計算並應用變換
    # cv2.estimateAffinePartial2D 返回變換矩陣 M 和一個內點掩碼 (inliers mask)
    M, _ = cv2.estimateAffinePartial2D(src, dst)

    # 檢查 M 是否成功計算
    if M is None:
        # 如果無法計算變換矩陣，記錄警告並返回原始圖像或引發錯誤
        # 這裡選擇記錄警告並返回 None，讓呼叫者處理
        # import logging # 如果需要記錄日誌，請在檔案頂部導入
        # logging.warning("Failed to estimate affine transform in face_align.")
        # 或者可以引發一個異常
        raise ValueError("Failed to estimate affine transform in face_align.")
        # return None # 或者返回 None

    # 應用仿射變換
    aligned = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)

    return aligned
