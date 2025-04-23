import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any  # Added Dict, Any

# --- Module Imports (Ensure these are accessible) ---
from face_detection import face_detect_bgr
from feature_extraction import feature_extract
from face_comparison import compare_face
from utils import detector, sess, db_path, settings  # 導入 settings 而不是 recognition_threshold
from database import table_exists  # 匯入表格檢查函數

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 常數 ---
RECOGNITION_INTERVAL = 3  # 每隔 N 幀進行一次辨識 (可調整)

# --- 輔助函數：在畫面上繪製結果 ---
def _draw_results_on_frame(frame: np.ndarray, positions: List[List[int]], display_texts: List[str]):
    """在畫面上繪製辨識結果的邊界框和文字。"""
    for i, pos in enumerate(positions):
        # 確保 pos 是有效的邊界框且對應的文字存在
        if isinstance(pos, (list, tuple)) and len(pos) == 4 and i < len(display_texts):
            try:
                x1, y1, x2, y2 = map(int, pos)  # 確保是整數
                text = display_texts[i]
                color = (0, 255, 0) if "Unknown" not in text and "Error" not in text else (0, 0, 255)  # BGR

                # 繪製矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 準備文字標籤
                label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # 計算文字背景框的 y 座標，確保它在圖像頂部邊界內
                y1_label_top = max(y1 - label_size[1] - 10, 0)
                # 計算文字基線的 y 座標
                y1_label_baseline = y1_label_top + label_size[1] + 7  # 調整基線位置

                # 繪製文字背景框
                cv2.rectangle(frame, (x1, y1_label_top),
                              (x1 + label_size[0], y1_label_top + label_size[1] + base_line), color, cv2.FILLED)
                # 繪製文字
                cv2.putText(frame, text, (x1, y1_label_baseline - base_line + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception as draw_err:
                logging.warning(f"Error drawing result for pos {pos} and text '{text}': {draw_err}")
                continue  # 跳過這個繪製錯誤，繼續處理下一個

# --- 輔助函數：處理單一畫面進行辨識 ---
def _process_frame_for_recognition(frame_bgr: np.ndarray, detector: Any, sess: Any, db_path: str) -> Tuple[List[List[int]], List[str]]:
    """
    偵測、提取特徵、比對人臉，並返回位置和顯示文字。
    使用 utils.settings['recognition_threshold']。
    """
    positions_list = []
    display_texts_list = []

    # 1. 偵測人臉和關鍵點
    frame_rgb, detections = face_detect_bgr(frame_bgr, detector)

    if frame_rgb is not None and detections:
        # 2. 提取人臉特徵
        positions, _, embeddings = feature_extract(frame_rgb, detections, sess)

        # 檢查 face_embeddings 表格是否存在
        embeddings_table_exists = table_exists(db_path, 'face_embeddings')
        if not embeddings_table_exists:
            logging.error(f"Database table 'face_embeddings' does not exist at {db_path}. Cannot perform comparison.")

        # 3. 比對每張偵測到的人臉
        for i, embedding in enumerate(embeddings):
            if embedding is None:
                continue  # 跳過無效嵌入

            name = "Unknown"
            distance = float('inf')

            if embeddings_table_exists:
                try:
                    current_threshold = settings.get('recognition_threshold', 0.6)  # 提供預設值
                    name, distance, _ = compare_face(embedding, current_threshold, db_path)
                except Exception as comp_err:
                    logging.error(f"Error during face comparison for face {i}: {comp_err}", exc_info=True)
                    name = "Error: Comparison Failed"
            else:
                name = "Error: DB Table Missing"

            if i < len(positions) and positions[i] is not None:
                pos = positions[i]
                display_text = name
                if name != 'Unknown' and "Error" not in name and distance != float('inf'):
                    try:
                        similarity_percentage = max(0.0, 100.0 * (1.0 - distance))
                        display_text = f"{name}: {similarity_percentage:.1f}%"
                    except TypeError:
                        pass  # 保持 display_text 為 name

                positions_list.append(pos)
                display_texts_list.append(display_text)

    return positions_list, display_texts_list

# --- 輔助函數：條件性更新辨識結果 ---
def _conditionally_update_recognition(frame_counter: int, frame_bgr: np.ndarray, detector: Any, sess: Any, db_path: str, last_positions: List[List[int]], last_display_texts: List[str]) -> Tuple[List[List[int]], List[str]]:
    """
    根據幀計數器決定是否執行辨識，並更新或清除儲存的辨識結果。
    使用 utils.settings['recognition_threshold']。

    :param frame_counter: 當前幀數。
    :param frame_bgr: 當前 BGR 畫面。
    :param detector: MediaPipe 偵測器。
    :param sess: ONNX 推理會話。
    :param db_path: 資料庫路徑。
    :param last_positions: 上次儲存的邊界框列表。
    :param last_display_texts: 上次儲存的顯示文字列表。
    :return: Tuple (updated_positions, updated_texts) - 更新後的邊界框和文字列表。
    """
    updated_positions = last_positions
    updated_texts = last_display_texts

    if frame_counter % RECOGNITION_INTERVAL == 0:
        positions, display_texts = _process_frame_for_recognition(
            frame_bgr, detector, sess, db_path
        )
        # 如果這次辨識有結果，則更新儲存的結果
        if positions:
            updated_positions = positions
            updated_texts = display_texts
        # 如果這次辨識沒有結果 (沒偵測到人臉或提取失敗)，則清除儲存的結果
        else:
            updated_positions = []
            updated_texts = []

    # 如果不是辨識幀，則直接返回上次的結果
    return updated_positions, updated_texts
