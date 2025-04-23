# filepath: d:\文件\作業\大四下\影像處理\face-recognition-main\face-recognition-web\app.py
import sys
import os
import cv2
import numpy as np
import base64
import json
import traceback
import math
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# --- Setup Python Path ---
# 將 src 目錄添加到 Python 路徑，以便匯入自訂模組
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Import Custom Modules ---
try:
    from face_recognition import _process_frame_for_recognition
    from utils import detector, sess, db_path, settings, update_setting
    from feature_extraction import feature_extract
    from face_detection import face_detect_bgr
    from database import (
        create_connection, create_table, table_exists,
        add_check_in_record, CHECK_IN_TABLE_SQL,
        USERS_TABLE_SQL,
        add_user, get_user_by_username,
        get_all_check_ins, get_check_ins_by_username,
        get_all_users, delete_user,
        FACE_EMBEDDING_TABLE_SQL,
        add_face_embedding
    )
except ImportError as e:
    # 關鍵模組匯入失敗，記錄錯誤並退出
    print(f"FATAL: Error importing core modules: {e}")
    print(f"Attempted import based on path: {src_path}")
    traceback.print_exc()
    sys.exit(1)

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_for_dev') # 建議在生產環境中設定環境變數

# --- Logging Configuration ---
# 在生產環境中，建議使用 INFO 或 WARNING 級別
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants and Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
DEFAULT_PHOTO = 'default_avatar.png'
CHECK_IN_COOLDOWN = timedelta(seconds=settings.get('check_in_cooldown_seconds', 5)) # 從設定檔讀取冷卻時間

# --- Global Variables ---
last_successful_check_in_time = None # 初始化全域變數

# --- Database Initialization ---
def initialize_database():
    """初始化資料庫，建立必要的表格。"""
    logging.info("Initializing database...")
    db_dir = os.path.dirname(db_path)
    try:
        # 確保資料庫目錄存在
        os.makedirs(db_dir, exist_ok=True)
        logging.info(f"Ensured database directory exists: {db_dir}")
    except OSError as e:
        logging.error(f"Error creating database directory {db_dir}: {e}", exc_info=True)
        # 如果目錄無法建立，後續操作很可能會失敗，但還是嘗試繼續

    conn = None # 初始化 conn
    try:
        conn = create_connection(db_path)
        if conn is None:
            logging.error(f"Failed to create database connection to {db_path}")
            return # 無法連線，直接返回

        # 檢查並建立表格
        tables_to_check = {
            'check_ins': CHECK_IN_TABLE_SQL,
            'users': USERS_TABLE_SQL,
            'face_embeddings': FACE_EMBEDDING_TABLE_SQL
        }
        for table_name, creation_sql in tables_to_check.items():
            create_table(conn, creation_sql) # 直接調用 create_table
            logging.info(f"Created '{table_name}' table.")
        else:
            logging.info(f"'{table_name}' table already exists.")

    except Exception as e:
        logging.error(f"Error during table creation or check: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed after initialization.")

initialize_database()
# --- End Database Initialization ---

# --- Decorators ---
def login_required(f):
    """裝飾器：要求使用者必須登入。"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('請先登入以存取此頁面。', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """裝飾器：要求使用者必須是管理員。"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('請先登入以存取此頁面。', 'warning')
            return redirect(url_for('login'))
        if not session.get('is_admin'):
            flash('您沒有權限存取此頁面。', 'danger')
            return redirect(url_for('index')) # 或導向使用者儀表板
        return f(*args, **kwargs)
    return decorated_function

# --- Helper Functions ---
def decode_base64_image(base64_string):
    """將 Base64 編碼的圖片字串解碼為 OpenCV 影像 (BGR)。"""
    if not base64_string or not base64_string.startswith('data:image'):
        logging.warning("Invalid base64 image string received.")
        return None
    try:
        img_str = base64_string.split(',', 1)[1]
        image_data = base64.b64decode(img_str)
        nparr = np.frombuffer(image_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            logging.warning("cv2.imdecode failed for base64 data.")
        return img_bgr
    except (IndexError, base64.binascii.Error, Exception) as e:
        logging.error(f"Error decoding base64 image: {e}", exc_info=True)
        return None

def calculate_distance(p1, p2):
    """計算兩個關鍵點之間的歐氏距離"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_yaw_ratio(landmarks):
    """根據臉部關鍵點計算 Yaw 比例。"""
    if len(landmarks) < 3:
        logging.warning("Not enough landmarks provided to calculate_yaw_ratio.")
        return None
    left_eye, right_eye, nose_tip = landmarks[0], landmarks[1], landmarks[2]
    dist_left_eye_nose = calculate_distance(left_eye, nose_tip)
    dist_right_eye_nose = calculate_distance(right_eye, nose_tip)
    if dist_right_eye_nose == 0:
        logging.warning("Distance between right eye and nose is zero in calculate_yaw_ratio.")
        return None
    return dist_left_eye_nose / dist_right_eye_nose

def save_profile_photo(username, img_bgr):
    """儲存使用者個人照片 (第一張註冊照片)。"""
    if img_bgr is None:
        logging.error(f"Cannot save profile photo for {username}: image is None.")
        return False, "無法儲存照片：圖片資料無效。"

    user_photo_dir = os.path.join(DATASET_FOLDER, username)
    photo_path = os.path.join(user_photo_dir, 'profile.jpg')

    try:
        os.makedirs(user_photo_dir, exist_ok=True)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        result, encoded_img = cv2.imencode('.jpg', img_bgr, encode_param)

        if result:
            with open(photo_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            logging.info(f"Successfully saved profile photo for {username} at {photo_path}.")
            return True, "個人照片儲存成功。"
        else:
            logging.error(f"cv2.imencode failed for photo {username}.")
            return False, "圖片編碼失敗，無法儲存個人照片。"
    except OSError as ose:
        logging.error(f"OS Error saving photo for {username} at {user_photo_dir}: {ose}", exc_info=True)
        return False, f"儲存個人照片時發生系統錯誤 (OSError): {ose}"
    except Exception as e:
        logging.error(f"Unexpected error saving photo for {username}: {e}", exc_info=True)
        return False, "儲存個人照片時發生未知錯誤。"

# --- Routes ---

@app.route('/')
def index():
    """顯示首頁。"""
    return render_template('home.html',
                           logged_in='user_id' in session,
                           username=session.get('username'),
                           is_admin=session.get('is_admin'))

@app.route('/live_checkin')
def live_checkin():
    """提供即時打卡頁面。"""
    return render_template('live_checkin.html')

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """接收影像幀，執行辨識並返回結果。"""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400

    frame_bgr = decode_base64_image(data['image'])
    if frame_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        positions, display_texts = _process_frame_for_recognition(
            frame_bgr, detector, sess, db_path
        )
        return jsonify({"positions": positions, "texts": display_texts})
    except Exception as e:
        logging.error(f"Error during recognition process: {e}", exc_info=True)
        return jsonify({"error": "Recognition process failed", "details": str(e)}), 500

@app.route('/check_in', methods=['POST'])
def check_in():
    """記錄打卡時間 (由前端觸發)。"""
    data = request.get_json()
    name = data.get('name')

    if not name or name == "Unknown" or "Error" in name:
        return jsonify({"status": "failed", "message": "Cannot check in invalid or Unknown user."}), 400

    # --- 全域冷卻檢查 ---
    global last_successful_check_in_time # <--- 在使用前聲明 global
    now = datetime.now()
    # 檢查變數是否存在且不為 None
    if last_successful_check_in_time is not None and (now - last_successful_check_in_time) < CHECK_IN_COOLDOWN:
         cooldown_remaining = (CHECK_IN_COOLDOWN - (now - last_successful_check_in_time)).total_seconds()
         logging.warning(f"Check-in attempt for {name} blocked due to cooldown. Remaining: {cooldown_remaining:.1f}s")
         return jsonify({"status": "failed", "message": f"操作過於頻繁，請稍候 {cooldown_remaining:.1f} 秒。"}), 429 # Too Many Requests
    # --- 結束冷卻檢查 ---

    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    check_in_type = data.get('type', 'in') # 預設為 'in'

    conn = None
    try:
        conn = create_connection(db_path)
        if conn is None:
            logging.error("Database connection failed for check-in.")
            return jsonify({"status": "failed", "message": "Database connection failed."}), 500

        record_id = add_check_in_record(conn, name, timestamp, check_in_type)
        if record_id:
            last_successful_check_in_time = now # 更新全域變數
            logging.info(f"Check-in recorded: ID={record_id}, Name={name}, Time={timestamp}, Type={check_in_type}")
            return jsonify({"status": "success", "message": f"{name} checked {check_in_type} at {timestamp}", "record_id": record_id})
        else:
            logging.error(f"Failed to write check-in record for {name}.")
            return jsonify({"status": "failed", "message": "Failed to write check-in record to database."}), 500
    except Exception as e:
        logging.error(f"Error during check-in database operation: {e}", exc_info=True)
        return jsonify({"status": "failed", "message": "Server error during check-in."}), 500
    finally:
        if conn:
            conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    """處理使用者登入。"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('請輸入使用者名稱和密碼。', 'warning')
            return redirect(url_for('login'))

        conn = None
        user = None
        try:
            conn = create_connection(db_path)
            if not conn:
                flash('資料庫連線失敗，請稍後再試。', 'danger')
                return render_template('login.html')

            user = get_user_by_username(conn, username)

        except Exception as e:
            logging.error(f"Error fetching user during login: {e}", exc_info=True)
            flash('登入時發生錯誤，請稍後再試。', 'danger')
        finally:
            if conn:
                conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            flash(f'歡迎回來, {user["username"]}!', 'success')
            return redirect(url_for('admin_dashboard') if user['is_admin'] else url_for('my_records'))
        else:
            flash('使用者名稱或密碼錯誤。', 'danger')
            # 保持在登入頁面，而不是重定向，這樣表單內容可以保留 (如果模板支持)
            return render_template('login.html')

    # GET 請求
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """處理新使用者註冊 (接收預先驗證過的 3 張照片)。"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        face_images_json = request.form.get('face_images_base64_json')

        # --- 基本欄位驗證 ---
        if not all([username, password, confirm_password]):
            flash('使用者名稱和密碼欄位皆為必填。', 'warning')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('兩次輸入的密碼不一致。', 'warning')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('密碼長度至少需要 6 個字元。', 'warning')
            return redirect(url_for('register'))
        # --- 結束基本欄位驗證 ---

        # --- 處理臉部影像 ---
        face_images_base64 = []
        if face_images_json:
            try:
                face_images_base64 = json.loads(face_images_json)
                if not isinstance(face_images_base64, list) or len(face_images_base64) != 3:
                    flash('臉部影像資料格式錯誤或數量不符 (需要 3 張)。', 'danger')
                    return redirect(url_for('register'))
            except json.JSONDecodeError:
                flash('臉部影像資料格式錯誤 (JSON 解析失敗)。', 'warning')
                return redirect(url_for('register'))
        else:
            flash('缺少臉部影像資料。', 'danger')
            return redirect(url_for('register'))
        # --- 結束處理臉部影像 ---

        logging.info(f"Processing registration for {username} with 3 validated images.")

        # --- 提取特徵 ---
        processed_embeddings = []
        first_image_bgr = None # 用於儲存個人照片
        for i, img_base64 in enumerate(face_images_base64):
            img_bgr = decode_base64_image(img_base64)
            if img_bgr is None:
                logging.error(f"Could not decode pre-validated image {i+1} for {username}.")
                flash(f'處理第 {i+1} 張已驗證的照片時發生解碼錯誤。', 'danger')
                return redirect(url_for('register'))

            if i == 0: first_image_bgr = img_bgr # 儲存第一張圖片

            try:
                img_rgb, detections = face_detect_bgr(img_bgr, detector)
                if not detections or len(detections) != 1:
                    logging.error(f"Detection error on pre-validated image {i+1} for {username}.")
                    flash(f'處理第 {i+1} 張已驗證的照片時發生偵測錯誤。', 'danger')
                    return redirect(url_for('register'))

                # feature_extract 需要 BGR 影像
                _, _, list_of_embeddings = feature_extract(img_bgr, detections, sess)

                if list_of_embeddings and list_of_embeddings[0] is not None:
                    processed_embeddings.append(list_of_embeddings[0])
                else:
                    logging.error(f"Failed to extract features from pre-validated image {i+1} for {username}.")
                    flash(f'處理第 {i+1} 張已驗證的照片時發生特徵提取錯誤。', 'danger')
                    return redirect(url_for('register'))
            except Exception as e:
                logging.error(f"Error processing pre-validated image {i+1} for {username}: {e}", exc_info=True)
                flash(f'處理第 {i+1} 張已驗證的照片時發生未知錯誤。', 'danger')
                return redirect(url_for('register'))
        # --- 結束特徵提取 ---

        if len(processed_embeddings) != 3:
            logging.error(f"Feature extraction count mismatch ({len(processed_embeddings)} != 3) for {username}.")
            flash('臉部特徵提取數量不符。', 'danger')
            return redirect(url_for('register'))

        # --- 儲存使用者和特徵 (使用事務) ---
        hashed_password = generate_password_hash(password)
        conn = None
        registration_successful = False
        try:
            conn = create_connection(db_path)
            if not conn:
                flash('資料庫連線失敗，無法註冊。', 'danger')
                return render_template('register.html')

            conn.execute("BEGIN TRANSACTION;") # 開始事務

            user_id = add_user(conn, username, hashed_password, is_admin=False)
            if not user_id:
                # add_user 內部應處理名稱重複等問題，這裡假設返回 None 表示失敗
                flash('註冊失敗，使用者名稱可能已被使用。', 'danger')
                conn.execute("ROLLBACK;") # 回滾事務
                return redirect(url_for('register'))

            for idx, embedding in enumerate(processed_embeddings):
                if not add_face_embedding(conn, username, embedding, source_image_path=f"registration_{username}_{idx}"):
                    logging.error(f"Error adding face embedding {idx} for {username}.")
                    flash('儲存臉部特徵時發生資料庫錯誤。', 'danger')
                    conn.execute("ROLLBACK;") # 回滾事務
                    return redirect(url_for('register'))

            conn.execute("COMMIT;") # 提交事務
            registration_successful = True
            logging.info(f"User {username} and embeddings committed to database.")

        except Exception as e:
            logging.error(f"Error during user/embedding saving transaction for {username}: {e}", exc_info=True)
            flash('註冊過程中發生資料庫錯誤。', 'danger')
            if conn:
                try: conn.execute("ROLLBACK;")
                except Exception as rb_err: logging.error(f"Error during rollback: {rb_err}")
            return redirect(url_for('register'))
        finally:
            if conn: conn.close()
        # --- 結束儲存使用者和特徵 ---

        # --- 儲存個人照片 (如果註冊成功) ---
        if registration_successful:
            photo_saved, photo_message = save_profile_photo(username, first_image_bgr)
            if photo_saved:
                flash(f'註冊成功！已新增臉部資料並儲存個人照片。您現在可以登入了。', 'success')
            else:
                # 即使照片儲存失敗，註冊本身是成功的
                flash(f'註冊成功！已新增臉部資料，但儲存個人照片時發生錯誤：{photo_message}', 'warning')
            return redirect(url_for('login'))
        else:
            # 如果註冊不成功 (理論上前面已處理並重定向，但作為保險)
            flash('註冊失敗，請重試。', 'danger')
            return redirect(url_for('register'))

    # GET 請求
    return render_template('register.html')

@app.route('/validate_registration_photo', methods=['POST'])
def validate_registration_photo():
    """接收單張註冊照片進行即時驗證。"""
    data = request.get_json()
    img_base64 = data.get('image_base64')
    accepted_ratios = data.get('accepted_ratios', []) # 提供預設空列表

    if not img_base64:
        return jsonify({"status": "error", "message": "缺少圖片資料。"}), 400

    img_bgr = decode_base64_image(img_base64)
    if img_bgr is None:
        return jsonify({"status": "error", "message": "無法解碼圖片。"}), 400

    try:
        # 臉部偵測
        img_rgb, detections = face_detect_bgr(img_bgr, detector)

        # 品質與數量驗證
        if not detections:
            return jsonify({"status": "error", "message": "未偵測到臉部。"}), 200
        if len(detections) > 1:
            return jsonify({"status": "error", "message": "偵測到多張臉部，請確保只有一張臉。"}), 200

        detection = detections[0]
        bbox = detection.get('bbox')
        landmarks = detection.get('landmarks')

        if not bbox or not landmarks:
             logging.error("Invalid detection format received from face_detect_bgr.")
             return jsonify({"status": "error", "message": "臉部偵測結果格式錯誤。"}), 500

        # 清晰度驗證
        (x1, y1, x2, y2) = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
        if x1 >= x2 or y1 >= y2:
             logging.warning(f"Invalid bounding box calculated: ({x1},{y1})-({x2},{y2})")
             return jsonify({"status": "error", "message": "無效的臉部邊界框。"}), 500 # 可能是伺服器端問題
        gray_face = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        blur_threshold_val = settings.get('blur_threshold', 100) # 使用設定檔的值，提供預設
        if laplacian_var < blur_threshold_val:
            return jsonify({"status": "error", "message": f"照片太模糊 (清晰度: {laplacian_var:.2f}，需 > {blur_threshold_val})。"}), 200

        # 角度差異驗證
        current_ratio = calculate_yaw_ratio(landmarks)
        if current_ratio is None:
            return jsonify({"status": "error", "message": "無法計算臉部角度比例。"}), 500

        ratio_diff_threshold_val = settings.get('ratio_diff_threshold', 0.2) # 使用設定檔的值，提供預設
        if accepted_ratios:
            min_rel_diff_found = float('inf')
            for accepted_ratio in accepted_ratios:
                # 避免除以零
                relative_difference = abs(current_ratio - accepted_ratio) / abs(accepted_ratio) if accepted_ratio != 0 else float('inf')
                min_rel_diff_found = min(min_rel_diff_found, relative_difference)

            if min_rel_diff_found <= ratio_diff_threshold_val:
                message = f"角度與已接受的照片太相似 (最小相對差異: {min_rel_diff_found:.2%}，需 > {ratio_diff_threshold_val:.2%})。"
                return jsonify({"status": "error", "message": message}), 200

        # 如果通過所有檢查
        return jsonify({
            "status": "success",
            "message": "照片合格！",
            "ratio": current_ratio
        }), 200

    except Exception as e:
        logging.error(f"Error validating registration photo: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "伺服器處理照片時發生錯誤。"}), 500

@app.route('/logout')
@login_required
def logout():
    """處理使用者登出。"""
    session.clear() # 清除所有 session 資料更徹底
    flash('您已成功登出。', 'info')
    return redirect(url_for('login'))

@app.route('/my_records')
@login_required
def my_records():
    """顯示當前登入使用者的打卡記錄。"""
    username = session.get('username')
    records = []
    conn = None
    try:
        conn = create_connection(db_path)
        if conn:
            records = get_check_ins_by_username(conn, username)
        else:
            flash('無法連接資料庫以查詢記錄。', 'danger')
    except Exception as e:
        logging.error(f"Error fetching records for {username}: {e}", exc_info=True)
        flash('查詢打卡記錄時發生錯誤。', 'danger')
    finally:
        if conn:
            conn.close()
    return render_template('my_records.html', records=records, username=username)

# --- Admin Routes ---

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """顯示管理員儀表板。"""
    return render_template('admin_dashboard.html', username=session.get('username'))

@app.route('/admin/records')
@admin_required
def admin_records():
    """顯示所有使用者的打卡記錄。"""
    records = []
    conn = None
    try:
        conn = create_connection(db_path)
        if conn:
            records = get_all_check_ins(conn)
        else:
            flash('無法連接資料庫以查詢記錄。', 'danger')
    except Exception as e:
        logging.error(f"Error fetching all records for admin: {e}", exc_info=True)
        flash('查詢所有打卡記錄時發生錯誤。', 'danger')
    finally:
        if conn:
            conn.close()
    return render_template('admin_records.html', records=records)

@app.route('/admin/users', methods=['GET', 'POST'])
@admin_required
def admin_users():
    """管理使用者帳號和臉部資料。"""
    conn = None # 初始化 conn
    try:
        conn = create_connection(db_path)
        if not conn:
            flash('資料庫連線失敗。', 'danger')
            return redirect(url_for('admin_dashboard'))

        if request.method == 'POST':
            action = request.form.get('action')
            # 將 POST 處理邏輯移到單獨的函數中以提高可讀性
            handle_admin_user_post(conn, request.form, request.files)
            # POST 處理後重定向以避免重新提交表單
            return redirect(url_for('admin_users'))

        # GET 請求：獲取使用者列表
        users = get_all_users(conn)
        return render_template('admin_users.html', users=users)

    except Exception as e:
        logging.error(f"Error in admin_users route: {e}", exc_info=True)
        flash('處理使用者管理請求時發生錯誤。', 'danger')
        return redirect(url_for('admin_dashboard')) # 出錯時重定向到儀表板
    finally:
        if conn:
            conn.close()

def handle_admin_user_post(conn, form_data, file_data):
    """處理 /admin/users 的 POST 請求邏輯。"""
    action = form_data.get('action')

    if action == 'add_user':
        username = form_data.get('username')
        password = form_data.get('password')
        is_admin = form_data.get('is_admin') == 'on'

        if not username or not password:
            flash('請輸入新使用者的名稱和密碼。', 'warning')
        elif len(password) < 6:
            flash('密碼長度至少需要 6 個字元。', 'warning')
        else:
            hashed_password = generate_password_hash(password)
            try:
                if add_user(conn, username, hashed_password, is_admin):
                    flash(f'使用者 {username} 新增成功。', 'success')
                else:
                    flash(f'新增使用者 {username} 失敗 (可能名稱已存在)。', 'danger')
            except Exception as e:
                logging.error(f"Error adding user {username} from admin: {e}", exc_info=True)
                flash('新增使用者時發生資料庫錯誤。', 'danger')

    elif action == 'add_face':
        username = form_data.get('username_for_face')
        image_file = file_data.get('face_image')

        if not username or not image_file or image_file.filename == '':
            flash('請選擇要添加臉部的用戶名並上傳照片。', 'warning')
            return # 提前返回

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not ('.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            flash('只接受 PNG, JPG, JPEG 格式的圖片。', 'warning')
            return # 提前返回

        try:
            filestr = image_file.read()
            nparr = np.frombuffer(filestr, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                flash('無法讀取上傳的圖片。', 'danger')
                return

            img_rgb, detections = face_detect_bgr(img_bgr, detector)

            if not detections:
                flash('在照片中未偵測到臉部。', 'warning')
            elif len(detections) > 1:
                flash('照片中偵測到多張臉，請上傳只有一張臉的照片。', 'warning')
            else:
                # 提取特徵
                _, _, list_of_embeddings = feature_extract(img_bgr, detections, sess)

                if list_of_embeddings and list_of_embeddings[0] is not None:
                    feature = list_of_embeddings[0]
                    source_path = image_file.filename

                    if add_face_embedding(conn, username, feature, source_path):
                        flash(f'成功為使用者 {username} 添加臉部特徵。', 'success')
                    else:
                        flash(f'為使用者 {username} 添加臉部特徵失敗 (資料庫錯誤)。', 'danger')
                else:
                    flash(f'無法從 {username} 的照片中提取臉部特徵。', 'warning')

        except Exception as e:
            logging.error(f"Error processing uploaded face image for {username}: {e}", exc_info=True)
            flash('處理上傳的臉部照片時發生錯誤。', 'danger')

@app.route('/admin/users/delete/<username>', methods=['POST'])
@admin_required
def delete_user_route(username):
    """處理刪除使用者請求。"""
    if username == session.get('username'):
        flash('不能刪除您自己的帳號。', 'warning')
        return redirect(url_for('admin_users'))

    conn = None
    deleted = False
    try:
        conn = create_connection(db_path)
        if not conn:
            flash('資料庫連線失敗，無法刪除使用者。', 'danger')
            return redirect(url_for('admin_users'))

        deleted = delete_user(conn, username) # delete_user 應處理相關聯的 embeddings

    except Exception as e:
        logging.error(f"Error deleting user {username} via route: {e}", exc_info=True)
        flash(f'刪除使用者 {username} 時發生錯誤。', 'danger')
    finally:
        if conn:
            conn.close()

    if deleted:
        flash(f'使用者 {username} 已成功刪除。', 'success')
    else:
        # 如果 delete_user 返回 False 但沒有異常，可能是使用者不存在
        flash(f'刪除使用者 {username} 失敗 (可能使用者不存在或有關聯資料)。', 'danger')

    return redirect(url_for('admin_users'))

@app.route('/user_photo/<username>')
def serve_user_photo(username):
    """提供指定使用者的個人照片，找不到則提供預設圖片。"""
    photo_filename = 'profile.jpg'
    user_photo_dir = os.path.join(DATASET_FOLDER, username)
    user_photo_path = os.path.join(user_photo_dir, photo_filename)

    if os.path.exists(user_photo_path):
        try:
            # 使用 send_from_directory 提供檔案，更安全
            return send_from_directory(user_photo_dir, photo_filename)
        except Exception as e:
            logging.error(f"Error serving photo for {username} from {user_photo_dir}: {e}")
            # 出錯則回退到預設圖片

    # 如果找不到使用者照片或發生錯誤，提供預設圖片
    try:
        return send_from_directory(app.static_folder, DEFAULT_PHOTO)
    except Exception as e:
        logging.error(f"Error serving default photo {DEFAULT_PHOTO} from {app.static_folder}: {e}")
        return "Default photo not found", 404 # 連預設圖片都出錯

@app.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def admin_settings():
    """顯示和更新應用程式設定。"""
    if request.method == 'POST':
        update_errors = []
        for key, value in request.form.items():
            if key in settings: # 只處理存在於 settings 中的鍵
                if not update_setting(key, value):
                    update_errors.append(key)
            else:
                logging.warning(f"Ignoring unknown setting key from form: {key}")

        if not update_errors:
            flash('設定已成功更新。請注意，某些更改可能需要重新啟動應用程式才能完全生效。', 'success')
        else:
            flash(f'部分設定更新失敗: {", ".join(update_errors)}。請檢查輸入的值和類型。', 'danger')
        # 更新後重定向回 GET 請求，避免重新提交
        return redirect(url_for('admin_settings'))

    # GET 請求
    # 傳遞 settings 的副本以避免模板意外修改原始字典
    return render_template('admin_settings.html', current_settings=settings.copy())

# --- Main Execution ---
if __name__ == '__main__':
    # 檢查核心元件是否已初始化
    if 'sess' not in globals() or sess is None:
        logging.critical("FATAL: ONNX session is not initialized. Exiting.")
        sys.exit(1)
    if 'detector' not in globals() or detector is None:
        logging.critical("FATAL: MediaPipe detector is not initialized. Exiting.")
        sys.exit(1)

    logging.info("Starting Flask development server...")
    # 注意：app.run() 主要用於開發。生產環境應使用 Gunicorn 或類似 WSGI 伺服器。
    # 預設不啟用 debug 和 reloader，不使用 adhoc SSL。
    # 如果需要 HTTPS 開發，建議使用 Gunicorn 配合 openssl 生成的憑證。
    app.run(host='0.0.0.0', port=5000)