import cv2
import numpy as np
from face_alignment import face_align
from sklearn.preprocessing import normalize
import logging # 導入 logging
from typing import List, Tuple, Dict, Any # 添加類型提示

# --- Logging Configuration ---
# 可以根據需要調整級別
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 特徵提取函數 ---
def feature_extract(img_rgb: np.ndarray, detections: List[Dict[str, Any]], sess: Any) -> Tuple[List[List[int]], List[List[List[int]]], List[np.ndarray]]:
    """
    從偵測到的人臉中提取特徵嵌入。
    :param img_rgb: 輸入的 RGB 圖像 (NumPy array)。
    :param detections: 包含人臉偵測結果的列表 (每個元素是一個字典，包含 'bbox' 和 'landmarks')。
    :param sess: ONNX 推理會話。
    :return: Tuple (list_of_positions, list_of_landmarks_5, list_of_embeddings)
             - list_of_positions: 每個偵測到人臉的邊界框 [x1, y1, x2, y2]。
             - list_of_landmarks_5: 每個偵測到人臉的 5 個關鍵點 [[x1,y1], ..., [x5,y5]]。
             - list_of_embeddings: 每個成功處理人臉的特徵嵌入 (NumPy array)。
    """
    list_of_embeddings = []
    list_of_landmarks_5 = []
    list_of_positions = []

    # 基本檢查
    if img_rgb is None or not detections:
        return [], [], []

    # 移除了 try...except 塊
    # 假設 ONNX session 總是有效且至少有一個輸入
    input_name = sess.get_inputs()[0].name

    for face_info in detections:
        # 移除了 try...except 塊
        # 假設 face_info 總是包含有效的 'bbox' 和 'landmarks'
        bbox = face_info['bbox']
        landmarks_5 = face_info['landmarks']

        face_position = [int(b) for b in bbox]
        list_of_positions.append(face_position)
        list_of_landmarks_5.append(landmarks_5)

        # 1. 臉部對齊
        # 假設 face_align 總是成功或其內部錯誤處理足夠
        aligned_face_rgb = face_align(img_rgb, landmarks_5)
        # 如果 face_align 可能返回 None，需要檢查
        if aligned_face_rgb is None:
             logging.warning("Face alignment returned None, skipping this face.") # 保留警告
             continue # 對齊失敗則跳過

        # 2. 預處理
        input_proc = np.transpose(aligned_face_rgb, (2, 0, 1)).astype(np.float32)
        input_blob = np.expand_dims(input_proc, axis=0)

        # 3. ONNX 推理
        # 假設 sess.run 總是成功
        prediction = sess.run(None, {input_name: input_blob})[0]

        # 4. 後處理：L2 標準化
        final_embedding = normalize(prediction).flatten()
        list_of_embeddings.append(final_embedding)

    # 返回收集到的有效結果
    return list_of_positions, list_of_landmarks_5, list_of_embeddings

