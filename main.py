# ===== Standard library =====
import os
import uuid
import json
import base64
import smtplib
from datetime import datetime
from email.message import EmailMessage
import secrets
import re
import dns.resolver
import pytz
import random
# ===== Third-party libraries =====
import bcrypt
import mysql.connector
import numpy as np
import cv2
import tensorflow as tf
import face_recognition
import logging
import mediapipe as mp
from keras.models import load_model
from fastapi import FastAPI, Query, Request, Form, status, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import HTTP_303_SEE_OTHER
from starlette.websockets import WebSocketState
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from dotenv import load_dotenv

from fastapi.staticfiles import StaticFiles

from insightface.app import FaceAnalysis
app_arcface = FaceAnalysis(providers=['CPUExecutionProvider'])
app_arcface.prepare(ctx_id=0, det_size=(160, 160))

# ===== Local application imports =====
from face.anti_spoof_api import SFASPredictor
from face_occlusion.occlusion_detector import OcclusionDetector

# ===== Logging setup =====
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(message)s", level=logging.INFO)

logger_register = logging.getLogger("face_register")
logger_register.setLevel(logging.INFO)

logger_verify = logging.getLogger("face_verify")
logger_verify.setLevel(logging.INFO)

# ===== Environment setup =====
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===== Database config =====
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
}
# ===== FastAPI App =====
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="testsecret123")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
serializer = URLSafeTimedSerializer(os.getenv("SECRET_KEY"))

# ===== ML Models =====
sfas = SFASPredictor()
occlusion_detector = OcclusionDetector()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)  

# FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

def format_thai_datetime(dt: datetime) -> str:
    months = [
        "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
        "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
    ]

    tz = pytz.timezone("Asia/Bangkok")
    dt = dt.astimezone(tz)

    day = dt.day
    month = months[dt.month - 1]
    year = dt.year + 543
    time_str = dt.strftime("%H:%M:%S")
    return f"{day} {month} {year} เวลา {time_str}"

EMAIL_REGEX = re.compile(
    r"^(?:[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]"
    r"|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@"
    r"(?:gmail\.com|yahoo\.com|hotmail\.com|outlook\.com|icloud\.com|live\.com|msn\.com|proton\.me"
    r"|[a-zA-Z0-9.-]+\.ac\.th|[a-zA-Z0-9.-]+\.edu"
    r"|[a-zA-Z0-9.-]+\.go\.th|[a-zA-Z0-9.-]+\.co\.th|[a-zA-Z0-9.-]+\.or\.th)$"
)

PUBLIC_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", 
                  "icloud.com", "live.com", "msn.com", "proton.me"}

def domain_valid(email: str) -> bool:
    domain = email.split("@")[1].lower()

    if domain in PUBLIC_DOMAINS:
        return True

    for suffix in (".ac.th", ".edu", ".go.th", ".co.th", ".or.th"):
        if domain.endswith(suffix):
            return True

    return False

def domain_exists(email: str) -> bool:
    domain = email.split("@")[1]
    resolver = dns.resolver.Resolver()
    resolver.nameservers = ['8.8.8.8', '8.8.4.4']

    try:
        answers = resolver.resolve(domain, 'MX')
        if len(answers) > 0:
            return True
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN,
            dns.resolver.Timeout, dns.resolver.NoNameservers):
        try:
            a_records = resolver.resolve(domain, 'A')
            if len(a_records) > 0:
                return True
        except Exception:
            try:
                aaaa_records = resolver.resolve(domain, 'AAAA')
                if len(aaaa_records) > 0:
                    return True
            except Exception:
                return False
    return False

@app.get("/check-email")
async def check_email(email: str = Query(...)):
    if not EMAIL_REGEX.match(email):
        return {"exists": True, "message": "รูปแบบอีเมลไม่ถูกต้อง"}

    if not domain_valid(email):
        return {"exists": True, "message": "โดเมนอีเมลไม่ถูกต้องหรือไม่รองรับ"}

    user = get_user_by_email(email)
    if user:
        return {"exists": True, "message": "อีเมลนี้ถูกใช้งานไปแล้ว"}

    return {"exists": False, "message": "สามารถใช้ได้"}

def generate_otp(length=6):
    """สร้าง OTP ตัวเลข 6 หลัก"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(length)])

def send_otp_email(email: str, otp: str):
    html_content = f"""
    <div style="background-color:#0f1a1c;padding:40px;font-family:Arial,sans-serif;color:#d1d1d1;text-align:center;">
        <h2 style="color:white;">Your OTP Code</h2>
        <p>Use the following OTP to verify your action:</p>
        <h1 style="color:#c6ff00;font-size:32px;">{otp}</h1>
        <p>This code will expire in 10 minutes.</p>
    </div>
    """
    msg = EmailMessage()
    msg["Subject"] = "Your OTP Code"
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = email
    msg.set_content(f"Your OTP is: {otp}")
    msg.add_alternative(html_content, subtype="html")
    send_email(email, "Your OTP Code", msg)

@app.post("/send-otp")
async def send_otp(request: Request, email: str = Form(...)):
    otp = generate_otp()

    request.session['otp'] = otp
    request.session['otp_email'] = email
    request.session['otp_created_at'] = datetime.utcnow().timestamp()

    send_otp_email(email, otp)
    return HTMLResponse("OTP sent successfully")

@app.get("/check-username")
async def check_username(username: str = Query(...)):
    user = get_user_by_username(username)
    if user:
        return {"exists": True, "message": "ชื่อบัญชีนี้ถูกใช้ไปแล้ว กรุณาลองใหม่อีกครั้ง"}
    return {"exists": False, "message": "สามารถใช้ได้"}

@app.post("/verify-otp")
async def verify_otp(request: Request):
    data = await request.json()
    email = data.get("email")
    otp = str(data.get("otp", ""))

    session_otp = request.session.get("otp")
    session_email = request.session.get("otp_email")
    created_at = request.session.get("otp_created_at")

    if not session_otp or not session_email or not created_at:
        return {"valid": False, "message": "OTP หมดอายุหรือยังไม่ได้ส่ง"}

    try:
        created_at_ts = float(created_at)
    except (TypeError, ValueError):
        return {"valid": False, "message": "OTP หมดอายุ กรุณาขอใหม่"}

    if datetime.utcnow().timestamp() - created_at_ts > 600:
        del request.session['otp']
        del request.session['otp_email']
        del request.session['otp_created_at']
        return {"valid": False, "message": "OTP หมดอายุ กรุณาขอใหม่"}

    if email != session_email or otp != str(session_otp):
        return {"valid": False, "message": "OTP ไม่ถูกต้อง"}

    del request.session['otp']
    del request.session['otp_email']
    del request.session['otp_created_at']

    return {"valid": True, "message": "OTP ถูกต้อง"}

# ===== Utility Functions =====
def execute_db(query: str, args=()):
    """Execute a database write operation (INSERT, UPDATE, DELETE)."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, args)
        conn.commit()
        return True
    except mysql.connector.Error as e:
        print(f"[DB WRITE ERROR] {e}")
        if conn and conn.is_connected():
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def query_db(query: str, args=(), one=False):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, args)
        result = cursor.fetchone() if one else cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        result = None if one else []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return result

def get_user_by_username(username: str):
    return query_db('SELECT * FROM users WHERE username=%s', (username,), one=True)

def get_user_by_id(user_id: int):
    return query_db('SELECT * FROM users WHERE user_id=%s', (user_id,), one=True)

def get_user_by_email(email: str):
    return query_db('SELECT * FROM users WHERE email=%s', (email,), one=True)

def get_user_by_token(token: str):
    query = """
        SELECT 
            t.api_token, 
            t.token_id,
            u.credit_count, 
            u.*
        FROM tokens t
        JOIN users u ON t.user_id = u.user_id
        WHERE t.api_token = %s
        LIMIT 1
    """
    return query_db(query, (token,), one=True) 

def get_token_by_user_id(user_id: int):
    query = "SELECT api_token FROM tokens WHERE user_id=%s ORDER BY token_id ASC"
    results = query_db(query, (user_id,))
    return [r['api_token'] for r in results] if results else []

# ใน main.py เพิ่มฟังก์ชันเหล่านี้ใกล้เคียงกับ get_user_by_username

def get_subuser_by_username(username: str):
    """Retrieves a subuser by their unique username."""
    # ใช้สำหรับตรวจสอบว่า username ถูกใช้ไปแล้วหรือไม่
    return query_db('SELECT subuser_id, user_id, username FROM subusers WHERE username=%s', (username,), one=True)

def get_subusers_by_user_id(user_id: int):
    """Retrieves all subusers registered under a main user."""
    query = """
        SELECT 
            s.subuser_id, 
            s.username, 
            s.token_id, 
            s.registered_at, 
            s.last_verified_at,
            t.api_token
        FROM subusers s
        LEFT JOIN tokens t ON s.token_id = t.token_id
        WHERE s.user_id=%s
        ORDER BY s.registered_at DESC
    """
    return query_db(query, (user_id,))

def get_subuser_by_username_and_token_id(username: str, token_id: int):
    """
    Retrieves a subuser by their username, limited to a specific token_id.
    """
    # ค้นหา Subuser ที่มี username ตรงกัน ภายใต้ token_id นี้เท่านั้น
    query = 'SELECT subuser_id, token_id, username FROM subusers WHERE username=%s AND token_id=%s'
    params = (username, token_id)
    return query_db(query, params, one=True)

# NEW API: Endpoint สำหรับลบ Subuser
@app.post("/subuser/delete")
async def delete_subuser_endpoint(request: Request, subuser_id: int = Form(...)):
    user_id = request.session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")

    # ตรวจสอบว่า subuser นั้นเป็นของ user ที่ล็อกอินอยู่จริง
    if execute_db('DELETE FROM subusers WHERE subuser_id=%s AND user_id=%s', (subuser_id, user_id)):
        request.session['flash'] = f"ลบใบหน้าผู้ใช้งาน (ID: {subuser_id}) สำเร็จ"
    else:
        request.session['flash'] = "ไม่สามารถลบใบหน้าผู้ใช้งานได้"
        
    return RedirectResponse("/account#face-management-tab", status_code=HTTP_303_SEE_OTHER)

# NEW API: Endpoint สำหรับแก้ไขชื่อผู้ใช้งาน Subuser 
@app.post("/subuser/update_username") 
async def update_subuser_username_endpoint(request: Request, subuser_id: int = Form(...), new_username: str = Form(...)): 
    user_id = request.session.get('user_id') 
    if not user_id: 
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in") 

    # *** เพิ่ม: ดึง token_id ของ Subuser ที่กำลังแก้ไขออกมา ***
    # สมมติว่ามีฟังก์ชัน get_subuser_token_id ที่สามารถดึง token_id จาก subuser_id
    subuser_info = query_db('SELECT token_id FROM subusers WHERE subuser_id=%s AND user_id=%s', (subuser_id, user_id), one=True)
    if not subuser_info:
        request.session['flash'] = "ไม่พบ Subuser หรือคุณไม่มีสิทธิ์แก้ไข"
        return RedirectResponse("/account#face-management-tab", status_code=HTTP_303_SEE_OTHER)
        
    current_token_id = subuser_info['token_id']

    # 1. ตรวจสอบความซ้ำซ้อนของชื่อผู้ใช้งานใหม่
    # *** แก้ไข: เปลี่ยนไปกรองด้วย token_id แทน user_id ***
    existing_subuser = get_subuser_by_username_and_token_id(new_username, current_token_id) 
    
    # เงื่อนไข: ถ้าพบ Subuser ที่มีชื่อซ้ำภายใต้ token_id เดียวกัน และ Subuser ที่พบนั้นไม่ใช่ subuser_id ปัจจุบัน
    if existing_subuser and existing_subuser['subuser_id'] != subuser_id: 
        request.session['flash'] = "ชื่อผู้ใช้งานนี้ถูกใช้แล้วโดย Subuser อื่น **ภายใต้ API Token เดียวกัน**" 
        return RedirectResponse("/account#face-management-tab", status_code=HTTP_303_SEE_OTHER) 

    # 2. อัปเดตชื่อผู้ใช้งาน
    if execute_db('UPDATE subusers SET username=%s WHERE subuser_id=%s AND user_id=%s', (new_username, subuser_id, user_id)): 
        request.session['flash'] = f"แก้ไขชื่อผู้ใช้งานเป็น **{new_username}** สำเร็จ" 
    else: 
        request.session['flash'] = "ไม่สามารถแก้ไขชื่อผู้ใช้งานได้ (อาจเป็นเพราะข้อมูลไม่ถูกต้อง)" 
        
    return RedirectResponse("/account#face-management-tab", status_code=HTTP_303_SEE_OTHER)

# ===== Face Occlusion / Frontal Checks =====
ALL_KEY_INDICES = [
    33, 133, 362, 263, 1, 13, 14,
    10, 152, 234, 454,
    61, 291, 78, 308
]

def get_face_and_hand_data(frame, results_face, results_hands):
    h, w, _ = frame.shape
    is_occluded_by_hands = False

    if results_face.multi_face_landmarks and results_hands.multi_hand_landmarks:
        landmarks = results_face.multi_face_landmarks[0].landmark
        
        xs_face = [int(lm.x * w) for lm in landmarks]
        ys_face = [int(lm.y * h) for lm in landmarks]
        face_x1, face_x2 = min(xs_face), max(xs_face)
        face_y1, face_y2 = min(ys_face), max(ys_face)
        
        for hand_landmarks in results_hands.multi_hand_landmarks:
            xs_hand = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys_hand = [int(lm.y * h) for lm in hand_landmarks.landmark]
            hand_x1, hand_x2 = min(xs_hand), max(xs_hand)
            hand_y1, hand_y2 = min(ys_hand), max(ys_hand)
            
            intersection_x1 = max(face_x1, hand_x1)
            intersection_y1 = max(face_y1, hand_y1)
            intersection_x2 = min(face_x2, hand_x2)
            intersection_y2 = min(face_y2, hand_y2)
            
            intersection_width = max(0, intersection_x2 - intersection_x1)
            intersection_height = max(0, intersection_y2 - intersection_y1)
            intersection_area = intersection_width * intersection_height
            
            face_area = (face_x2 - face_x1) * (face_y2 - face_y1)
            hand_area = (hand_x2 - hand_x1) * (hand_y2 - hand_y1)
            
            union_area = face_area + hand_area - intersection_area
            
            if union_area > 0:
                iou = intersection_area / union_area
                if iou > 0.15:
                    is_occluded_by_hands = True
                    break
    
    return is_occluded_by_hands

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def is_face_frontal(frame, results_face, yaw_thresh=15, pitch_min=0.3, pitch_max=0.5):
    h, w, _ = frame.shape

    if not results_face.multi_face_landmarks:
        return False

    lm = results_face.multi_face_landmarks[0].landmark

    left_eye = lm[33]
    right_eye = lm[263]
    nose = lm[1]
    chin = lm[152]

    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
    nx, ny = int(nose.x * w), int(nose.y * h)
    cx, cy = int(chin.x * w), int(chin.y * h)

    # --- Yaw (ซ้าย/ขวา) ---
    eye_mid_x = (lx + rx) / 2
    nose_offset = nx - eye_mid_x
    yaw_ok = abs(nose_offset) < yaw_thresh

    # --- Pitch (ก้ม/เงย) ---
    eye_mid_y = (ly + ry) / 2
    vertical_ratio = (ny - eye_mid_y) / (cy - eye_mid_y)
    pitch_ok = pitch_min < vertical_ratio < pitch_max

    return yaw_ok and pitch_ok

# ===== Face Processing & Embedding =====
def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm

def extract_face_from_frame(rgb_frame: np.ndarray, detector, image_size=(160, 160)) -> tuple[np.ndarray|None, str|None]:
    results = detector.process(rgb_frame)
    if not results.detections:
        return None, "ไม่พบใบหน้า"

    detection = results.detections[0]
    ih, iw, _ = rgb_frame.shape

    bbox = detection.location_data.relative_bounding_box
    x1 = max(0, min(iw - 1, int(bbox.xmin * iw)))
    y1 = max(0, min(ih - 1, int(bbox.ymin * ih)))
    x2 = max(0, min(iw - 1, x1 + int(bbox.width * iw)))
    y2 = max(0, min(ih - 1, y1 + int(bbox.height * ih)))

    keypoints = detection.location_data.relative_keypoints
    if keypoints and len(keypoints) >= 2:
        left_eye = (int(keypoints[1].x * iw), int(keypoints[1].y * ih))
        right_eye = (int(keypoints[0].x * iw), int(keypoints[0].y * ih))
    else:
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        left_eye = right_eye = (cx, cy)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    aligned = cv2.warpAffine(rgb_frame, M, (iw, ih))

    face = aligned[y1:y2, x1:x2]
    if face.size == 0:
        return None, "ไม่พบใบหน้า"

    gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray_face)
    if brightness < 50:
        return None, "แสงไม่เพียงพอสำหรับการตรวจจับใบหน้า"
    elif brightness > 200:
        return None, "แสงบนหน้ามากเกินไป"

    face_yuv = cv2.cvtColor(face, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_yuv[:, :, 0] = clahe.apply(face_yuv[:, :, 0])
    face = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)

    return cv2.resize(face, image_size), None

def get_embedding(face_pixels: np.ndarray) -> np.ndarray:
    faces = app_arcface.get(face_pixels)
    if len(faces) == 0:
        return None
    return l2_normalize(faces[0].embedding)

def compare_embedding(embedding: np.ndarray, current_token_id: int, threshold=0.8):
    known_embeddings, known_labels, known_token_ids = load_known_embeddings_from_db()
    
    if known_embeddings.size == 0:
        return False, None, None

    filter_indices = [i for i, tid in enumerate(known_token_ids) if tid == current_token_id]
    
    if not filter_indices:
        return False, None, None

    filtered_embeddings = known_embeddings[filter_indices]
    filtered_labels = [known_labels[i] for i in filter_indices]

    distances = np.linalg.norm(filtered_embeddings - embedding, axis=1)
    min_idx = np.argmin(distances)
    
    if distances[min_idx] < threshold:
        return True, filtered_labels[min_idx], distances[min_idx]

    return False, None, None

def process_face_image(base64_data: str, detector, sfas):
    try:
        header, encoded = base64_data.split(",", 1) if "," in base64_data else ("", base64_data)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None, None, "Failed to decode image"
    except Exception as e:
        return None, None, f"Image decode error: {e}"

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    results = detector.process(frame_rgb)
    if not results.detections:
        return None, None, "No face detected"

    if len(results.detections) > 1:
        return None, None, f"Multiple faces detected ({len(results.detections)})"

    face, warning = extract_face_from_frame(frame_rgb, detector)
    if face is None:
        return None, None, warning

    face_pixels = face.astype('float32') / 255.0
    mean, std = face_pixels.mean(), face_pixels.std() + 1e-8
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)

    try:
        faces = app_arcface.get(frame_rgb)
        if not faces:
            return None, None, "No face detected"
        embedding = l2_normalize(faces[0].embedding)
    except Exception as e:
        return None, None, f"Embedding generation error: {e}"

    try:
        is_live, real_confidence, spoof_confidence = sfas.predict_live(frame_bgr)
    except Exception as e:
        return None, None, f"Liveness check error: {e}"

    result = {
        "frame": frame_bgr,
        "embedding": embedding,
        "is_live": is_live,
        "real_confidence": real_confidence,
        "spoof_confidence": spoof_confidence
    }

    return result, face, None

def load_known_embeddings_from_db():
    embeddings = []
    labels = []
    # เปลี่ยนชื่อตัวแปรที่คืนค่าเป็น token_ids
    known_token_ids = [] 
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # SQL Query: ดึง subuser_id, face_embedding, และ token_id
        cursor.execute("""
            SELECT subuser_id, face_embedding, token_id
            FROM subusers
            WHERE face_embedding IS NOT NULL AND face_embedding != ''
        """)
        records = cursor.fetchall()
    except mysql.connector.Error as e:
        print(f"[DB ERROR] {e}")
        return np.array([]), [], []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            
    for subuser_id, embedding_json, current_token_id_from_db in records: 
        if not embedding_json:
            continue
        try:
            embedding = np.array(json.loads(embedding_json), dtype=np.float32)
            labels.append(subuser_id)
            # เก็บค่า token_id
            known_token_ids.append(current_token_id_from_db) 
            embeddings.append(embedding)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[WARNING] Invalid embedding JSON for subuser '{subuser_id}': {e}")
            continue
            
    if embeddings:
        # คืนค่า embeddings, labels, และ token_ids
        return np.stack(embeddings), labels, known_token_ids
    return np.array([]), [], []

# ===== Email =====
def send_email(email: str, subject: str, message: EmailMessage):
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(message)
    except Exception as e:
        print(f"Error sending email: {e}")

def generate_token_link(email: str, user_id: int, salt: str, path: str):
    token_data = {"email": email, "user_id": user_id}
    token = serializer.dumps(token_data, salt=salt)
    base_url = os.getenv('BASE_URL', 'http://localhost:8000')
    return f"{base_url}/{path}/{token}"

def send_forgot_password_email(email: str, user_id: int, salt: str = "password-reset", path: str = "reset-password"):
    link = generate_token_link(email, user_id, salt, path)

    html_content = f"""
    <div style="background-color:#0f1a1c;padding:40px;font-family:Arial,sans-serif;color:#d1d1d1;text-align:center;">
        <h2 style="color:white;">Reset Your Password</h2>
        <p><strong>Hi {email.split('@')[0]},</strong></p>
        <p>We received a request to reset your password.</p>
        <p>If this was you, click the button below to proceed:</p>
        <a href="{link}" style="display:inline-block;background-color:#c6ff00;color:black;font-weight:bold;padding:14px 24px;text-decoration:none;border-radius:8px;margin-top:20px;">
            Reset Password
        </a>
        <p style="margin-top:20px;">If you didn't request this, please ignore this email.</p>
        <p style="font-size:12px;">Need help? Contact us at <a href="mailto:support@example.com" style="color:#c6ff00;">support@example.com</a></p>
    </div>
    """

    msg = EmailMessage()
    msg["Subject"] = "Reset Your Password"
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = email
    msg.set_content(f"Click to reset your password:\n{link}")
    msg.add_alternative(html_content, subtype="html")

    send_email(email, "Reset Your Password", msg)

# ===== Routes (HTTP) =====
@app.post("/account/create_api_token")
async def create_api_token(request: Request):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    user = get_user_by_username(username)
    if not user:
        request.session["flash"] = "ไม่พบผู้ใช้"
        return RedirectResponse(url="/account", status_code=status.HTTP_303_SEE_OTHER)

    user_id = user['user_id']
    # ดึงค่า token_count ปัจจุบันจาก user (ซึ่งถูกดึงมาจาก get_user_by_username)
    token_count = user.get('token_count', 0) 

    if token_count <= 0:
        request.session["flash"] = "คุณมี API Token ครบตามจำนวนที่กำหนดแล้ว (0 สิทธิ์คงเหลือ)"
        return RedirectResponse(url="/account", status_code=status.HTTP_303_SEE_OTHER)

    api_token = str(uuid.uuid4())

    # Transaction: Insert token and decrement token_count (-1)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        # 1. Insert new token
        c.execute(
            "INSERT INTO tokens (user_id, api_token) VALUES (%s, %s)",
            (user_id, api_token)
        )

        # 2. Decrement token_count (-1)
        c.execute(
            "UPDATE users SET token_count = token_count - 1 WHERE user_id = %s",
            (user_id,)
        )
        
        conn.commit()
        request.session["flash"] = f"สร้าง API Token ใหม่สำเร็จ: {api_token[:8]}..."
        
    except mysql.connector.Error as e:
        conn.rollback()
        print(f"[DB ERROR] Failed to create token: {e}")
        request.session["flash"] = "เกิดข้อผิดพลาดในการสร้าง API Token"
    finally:
        c.close()
        conn.close()

    return RedirectResponse(url="/account", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/account/delete_api_token")
async def delete_api_token(request: Request, api_token: str = Form(...)):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    user = get_user_by_username(username)
    if not user:
        request.session["flash"] = "ไม่พบผู้ใช้"
        return RedirectResponse(url="/account", status_code=status.HTTP_303_SEE_OTHER)

    user_id = user['user_id']
    
    # Transaction: Delete token and increment token_count (+1)
    conn = get_db_connection()
    c = conn.cursor()
    try:
        # 1. Delete the token (Ensure it belongs to the user)
        c.execute(
            "DELETE FROM tokens WHERE user_id = %s AND api_token = %s",
            (user_id, api_token)
        )
        
        if c.rowcount == 0:
            conn.rollback()
            request.session["flash"] = "ไม่พบ API Token นี้ หรือไม่ใช่ Token ของคุณ"
            return RedirectResponse(url="/account", status_code=status.HTTP_303_SEE_OTHER)

        # 2. Increment token_count (+1)
        c.execute(
            "UPDATE users SET token_count = token_count + 1 WHERE user_id = %s",
            (user_id,)
        )
        
        conn.commit()
        request.session["flash"] = f"ลบ API Token สำเร็จ"
        
    except mysql.connector.Error as e:
        conn.rollback()
        print(f"[DB ERROR] Failed to delete token: {e}")
        request.session["flash"] = "เกิดข้อผิดพลาดในการลบ API Token"
    finally:
        c.close()
        conn.close()

    return RedirectResponse(url="/account", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    msg = request.query_params.get("msg")
    message = None
    if msg == "password_changed":
        message = "เปลี่ยนรหัสผ่านสำเร็จ กรุณาเข้าสู่ระบบใหม่"
    return templates.TemplateResponse("login.html", {"request": request, "message": message})

@app.post("/auth/login", response_class=HTMLResponse)
async def login(request: Request, identifier: str = Form(...), password: str = Form(...)):
    user = get_user_by_username(identifier)
    if not user:
        user = get_user_by_email(identifier)

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "บัญชีผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"}
        )
    
    if password != "face_verified_password":
        if not bcrypt.checkpw(password.encode(), user['password'].encode()):
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "รหัสผ่านไม่ถูกต้อง"}
            )

    # ✅ เพิ่มบรรทัดนี้
    request.session['user_id'] = user['user_id']
    request.session['username'] = user['username']

    return RedirectResponse(url="/api", status_code=303)

@app.get("/api/face_authentication", response_class=HTMLResponse)
async def face_authentication(request: Request):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=303)

    user = get_user_by_username(username)
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    email = user.get("email")
    tokens_list = get_token_by_user_id(user["user_id"])
    credit_count = user.get("credit_count", 0)

    return templates.TemplateResponse("face_authentication.html", {
        "request": request,
        "user": username,
        "email": email,
        "activated": user.get("activated", 0),
        "tokens": tokens_list,
        "credit_count": credit_count
    })

@app.get("/api", response_class=HTMLResponse)
async def api_page(request: Request):
    username = request.session.get("username")

    context = {
        "request": request,
        "user": "Guest",
        "email": None,
        "credit_count": 0,
    }

    if username:
        user = get_user_by_username(username)
        if user:
            user_id = user["user_id"]
            
            context["user"] = username
            context["email"] = user.get("email")
            
            all_tokens = get_token_by_user_id(user_id) 
            
            credit_count = 0
            if all_tokens:
                main_api_token = all_tokens[0]
                
                token_info = get_user_by_token(main_api_token) 
                credit_count = token_info.get("credit_count") if token_info else 0
            
            context["credit_count"] = credit_count
            
    return templates.TemplateResponse("api.html", context)

# @app.get("/account", response_class=HTMLResponse)
# async def account(request: Request):
#     username = request.session.get("username")
#     if not username:
#         return RedirectResponse(url="/login", status_code=303)
#     user = query_db("SELECT * FROM users WHERE username=%s", (username,), one=True)
#     if not user:
#         return HTMLResponse("User not found", status_code=404)
#     user_id = user['user_id']
#     all_tokens = get_token_by_user_id(user_id)
#     main_api_token = all_tokens[0] if all_tokens else None
#     credit_count = 0
#     if main_api_token:
#         token_info = get_user_by_token(main_api_token)
#         credit_count = token_info.get('credit_count') if token_info else 0
#     final_credit_count = credit_count if user['activated'] else 0
#     return templates.TemplateResponse("account.html", {
#         "request": request,
#         "user": user["username"],
#         "email": user["email"],
#         "activated": user["activated"],
#         "credit_count": final_credit_count,
#         "tokens": all_tokens,
#         "token": main_api_token,
#         # เพิ่ม token_count เข้าไปใน context
#         "token_count": user.get('token_count', 0), 
#     })

@app.get("/account", response_class=HTMLResponse)
async def account(request: Request):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=303)
    user = query_db("SELECT * FROM users WHERE username=%s", (username,), one=True)
    if not user:
        return HTMLResponse("User not found", status_code=404)
    user_id = user['user_id']
    all_tokens = get_token_by_user_id(user_id)
    main_api_token = all_tokens[0] if all_tokens else None
    credit_count = 0
    if main_api_token:
        token_info = get_user_by_token(main_api_token)
        credit_count = token_info.get('credit_count') if token_info else 0
    final_credit_count = credit_count if user['activated'] else 0

    # NEW LOGIC: Fetch and format subusers data
    subusers_data = get_subusers_by_user_id(user_id)
    for subuser in subusers_data:
        if subuser['registered_at']:
            subuser['registered_at_formatted'] = format_thai_datetime(subuser['registered_at'])
        if subuser['last_verified_at']:
            subuser['last_verified_at_formatted'] = format_thai_datetime(subuser['last_verified_at'])
        else:
            subuser['last_verified_at_formatted'] = "-"
            
    # Preserve original context and add 'subusers'
    return templates.TemplateResponse("account.html", {
        "request": request,
        "user": user["username"],
        "email": user["email"],
        "activated": user["activated"],
        "credit_count": final_credit_count,
        "tokens": all_tokens,
        "token": main_api_token,
        # เพิ่ม token_count เข้าไปใน context
        "token_count": user.get('token_count', 0), 
        # NEW: Add subusers data
        "subusers": subusers_data,
    })

@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/auth/register")
async def register(request: Request,username: str = Form(...),password: str = Form(...),email: str = Form(...),):

    if query_db('SELECT * FROM users WHERE username=%s', (username,), one=True):
        request.session["flash"] = "ผู้ใช้นี้มีอยู่แล้ว"
        return RedirectResponse("/register", status_code=303)

    if query_db('SELECT * FROM users WHERE email=%s', (email,), one=True):
        request.session["flash"] = "อีเมลนี้มีอยู่แล้ว"
        return RedirectResponse("/register", status_code=303)

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO users (username, password, email, activated, credit_count, token_count) VALUES (%s, %s, %s, 1, 10, 3)',
            (username, hashed_pw.decode('utf-8'), email)
        )
        user_id = c.lastrowid

        for _ in range(2):
            api_token = str(uuid.uuid4())
            c.execute(
                "INSERT INTO tokens (user_id, api_token) VALUES (%s, %s)",
                (user_id, api_token)
            )

        conn.commit()

    finally:
        c.close()
        conn.close()

    return templates.TemplateResponse("register_success.html",{"request": request},status_code=201)

async def get_current_user(request: Request):
    username = request.session.get("username")
    if not username:
        return None
    user = get_user_by_username(username)
    return user

CREDIT_PACKAGES = {
    50: 100,
    150: 270,
    300: 450,
}

def get_placeholder_qr_base64():
    try:
        with open("static/placeholder_qr.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        print("!!! WARNING: static/placeholder_qr.png not found. QR code will not be displayed.")
        return ""


@app.get("/buy_credits", response_class=HTMLResponse)
async def buy_credits_page(request: Request, user: dict = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("buy_credits.html", {"request": request, "user": user})


@app.post("/create_payment", response_class=HTMLResponse)
async def create_payment(request: Request, credit_package: str = Form(...), user: dict = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    try:
        credit_amount = int(credit_package)
        if credit_amount not in CREDIT_PACKAGES:
            raise ValueError("Invalid package")
    except ValueError:
        return RedirectResponse(url="/buy_credits", status_code=status.HTTP_303_SEE_OTHER)

    total_price = CREDIT_PACKAGES[credit_amount]
    order_id = f"MORRIS-{uuid.uuid4().hex[:8].upper()}"

    request.session["payment_order"] = {
        "order_id": order_id,
        "credit_amount": credit_amount,
        "status": "PENDING",
        "poll_count": 0
    }
    
    qr_code_base64 = get_placeholder_qr_base64()

    return templates.TemplateResponse("payments.html", {
        "request": request,
        "credit_count": user.get("credit_count", 0),
        "credit_amount": credit_amount,
        "total_price": f"{total_price:,.2f}",
        "qr_code_base64": qr_code_base64,
        "order_id": order_id,
        "current_status": "PENDING"
    })

@app.get("/payment_status/{order_id}")
async def check_payment_status(request: Request, order_id: str):
    order_info = request.session.get("payment_order")

    if not order_info or order_info.get("order_id") != order_id:
        return {"status": "NOT_FOUND"}

    poll_count = order_info.get("poll_count", 0) + 1
    request.session["payment_order"]["poll_count"] = poll_count

    if poll_count >= 3:
        request.session["payment_order"]["status"] = "PAID"
        return {"status": "PAID"}

    return {"status": "PENDING"}


@app.get("/payment_success", response_class=HTMLResponse)
async def payment_success(request: Request, user: dict = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        
    order_info = request.session.get("payment_order")

    if not order_info or order_info.get("status") != "PAID":
        return RedirectResponse(url="/account", status_code=status.HTTP_303_SEE_OTHER)

    credit_added = order_info["credit_amount"]
    user_id = user["user_id"]
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET credit_count = credit_count + %s WHERE user_id = %s",
            (credit_added, user_id)
        )
        conn.commit()
        
        cursor.execute("SELECT credit_count FROM users WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        new_total_credits = result[0] if result else user.get("credit_count", 0) + credit_added
        
    except Exception as e:
        print(f"Error updating credits for user {user_id}: {e}")
        new_total_credits = "Error"
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

    del request.session["payment_order"]

    return templates.TemplateResponse("payment_success.html", {
        "request": request,
        "credit_added": credit_added,
        "new_total_credits": new_total_credits
    })

@app.get("/forgot-password")
async def forgot_password_page(request: Request):
    flash = request.session.pop("flash", None)
    return templates.TemplateResponse("forgot_password.html", {"request": request, "flash": flash})

@app.post("/forgot-password")
async def forgot_password(request: Request, identifier: str = Form(...)):
    user = get_user_by_email(identifier) or get_user_by_username(identifier)
    
    if user:
        send_forgot_password_email(user["email"], user["user_id"])
        request.session["flash"] = {
            "type": "success",
            "message": "A reset link has been sent to your email."
        }
    else:
        request.session["flash"] = {"type": "error","message": "Please enter a valid account."}

    return RedirectResponse(url="/forgot-password", status_code=303)

@app.get("/reset-password/{token}", response_class=HTMLResponse)
async def reset_password_form(request: Request, token: str):
    try:
        serializer.loads(token, salt="password-reset", max_age=1800)
    except SignatureExpired:
        return HTMLResponse("The reset link has expired.")
    except BadSignature:
        return HTMLResponse("Invalid reset link.")
    return templates.TemplateResponse("reset_password.html", {"request": request, "token": token})

@app.post("/reset-password/{token}", response_class=HTMLResponse)
async def reset_password(request: Request, token: str, password: str = Form(...)):
    message = None
    try:
        email = serializer.loads(token, salt="password-reset", max_age=1800)
        if isinstance(email, dict):
            email = email.get("email")
        
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute("SELECT password FROM users WHERE email=%s", (email,))
        row = c.fetchone()
        if not row:
            c.close()
            conn.close()
            return templates.TemplateResponse("reset_invalid.html", {"request": request})
        
        old_hashed_pw = row[0].encode('utf-8')

        if bcrypt.checkpw(password.encode('utf-8'), old_hashed_pw):
            message = "รหัสผ่านใหม่เหมือนรหัสปัจจุบัน กรุณาใช้รหัสผ่านอื่น"
            c.close()
            conn.close()
            return templates.TemplateResponse("reset_password.html", {
                "request": request,
                "token": token,
                "message": message
            })
        
        hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        c.execute('UPDATE users SET password=%s WHERE email=%s', (hashed_pw.decode('utf-8'), email))
        conn.commit()
        c.close()
        conn.close()

        return templates.TemplateResponse("login.html",
        {
        "request": request,
        "message": "ตั้งรหัสผ่านใหม่เรียบร้อยแล้ว กรุณาเข้าสู่ระบบด้วยรหัสผ่านใหม่"
        })

    except SignatureExpired:
        return templates.TemplateResponse("reset_expired.html", {"request": request})
    except BadSignature:
        return templates.TemplateResponse("reset_invalid.html", {"request": request})


@app.get("/change_password", response_class=HTMLResponse)
async def change_password_form(request: Request, user=Depends(get_current_user)):
    return templates.TemplateResponse("change_password.html", {"request": request, "user": user})

@app.post("/change_password")
async def change_password(
    request: Request,
    old_password: str = Form(...),
    new_password: str = Form(...),
    confirm_new_password: str = Form(...),
    user=Depends(get_current_user)
):
    
    if new_password != confirm_new_password:
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "user": user,
            "error": "รหัสผ่านใหม่กับยืนยันรหัสผ่านไม่ตรงกัน"
        })

    if new_password == old_password:
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "user": user,
            "error": "รหัสผ่านใหม่ต้องไม่ซ้ำกับรหัสผ่านเก่า"
        })

    if not bcrypt.checkpw(old_password.encode('utf-8'), user['password'].encode('utf-8')):
        return templates.TemplateResponse("change_password.html", {
            "request": request,
            "user": user,
            "error": "รหัสผ่านเก่าไม่ถูกต้อง"
        })

    hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET password=%s WHERE user_id=%s", (hashed_pw, user['user_id']))
    conn.commit()
    print("Password updated in DB for user_id:", user['user_id'])
    c.close()
    conn.close()

    return RedirectResponse(url="/logout?msg=password_changed", status_code=HTTP_303_SEE_OTHER)

@app.websocket("/api/ws/face-verify/")
async def websocket_verify(websocket: WebSocket):
    closed = False

    async def safe_close():
        nonlocal closed
        if not closed and websocket.client_state == WebSocketState.CONNECTED:
            closed = True
            try:
                await websocket.close()
            except Exception:
                pass
            logger_verify.info("WebSocket connection closed safely")

    try:
        await websocket.accept()
        init_msg = await websocket.receive_text()
        init_data = json.loads(init_msg)
        token = init_data.get("token")

        if not token:
            await safe_close()
            return

        user = get_user_by_token(token)
        if not user:
            await websocket.send_json({"body": {"success": False, "message": "ไม่พบผู้ใช้"}})
            await safe_close()
            return

        user_id = user["user_id"]
        
        current_token_id = None
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute(
                "SELECT token_id FROM tokens WHERE api_token = %s",
                (token,)
            )
            token_row = c.fetchone()
            current_token_id = token_row[0] if token_row else None
            c.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to fetch current_token_id: {e}")
        
        if current_token_id is None:
            await websocket.send_json({"body": {"success": False, "message": "ไม่พบ API Token หรือ token_id"}})
            await safe_close()
            return

    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        await safe_close()
        return

    async def send_face_response(success, message, user_id=None, subuser_id=None, confidence=None,
                                     spoof_confidence=None,faces_detected=1):
        nonlocal closed
        if closed or websocket.client_state != WebSocketState.CONNECTED:
            return

        if confidence is not None and spoof_confidence is None:
            spoof_confidence = 1 - confidence

        response = {
            "body": {
                "request_id": str(uuid.uuid4()),
                "user_id": user_id if user_id else None,
                "subuser_id": subuser_id if subuser_id else None,
                "success": success,
                "message": message,
                "confidence": round(float(confidence), 2) if confidence is not None else None,
                "spoof_confidence": round(float(spoof_confidence), 2) if spoof_confidence is not None else None,
                "faces_detected": faces_detected,
                "timestamp": format_thai_datetime(datetime.now(pytz.timezone("Asia/Bangkok"))),
            }
        }
        try:
            await websocket.send_json(response)
            logger_verify.info(json.dumps(response, indent=2, ensure_ascii=False))
        except Exception as e:
            closed = True
            logger.error(f"Failed to send JSON: {e}")

    async def log_face_result(matched, label, dist, is_live, real_confidence, spoof_confidence, occluded, face, is_occluded_by_hands, face_frontal):
        if is_occluded_by_hands or occluded or face is None:
            reason = "กรุณาเอามือออกจากใบหน้า" if is_occluded_by_hands else "กรุณาเอาสิ่งของออกจากใบหน้า"
            await send_face_response(False,reason,confidence=real_confidence,spoof_confidence=spoof_confidence)
        elif not face_frontal:
            await send_face_response(False, "กรุณาหันใบหน้าตรง", confidence=real_confidence, spoof_confidence=spoof_confidence)
        elif not is_live:
            await send_face_response(False, "ตรวจพบการปลอมใบหน้า", confidence=real_confidence, spoof_confidence=spoof_confidence)
        elif matched and is_live:
            await send_face_response(True, "ยืนยันตัวตนสำเร็จ", user_id=user_id, subuser_id=label, confidence=real_confidence, spoof_confidence=None)

            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE subusers SET last_verified_at = NOW() WHERE subuser_id = %s",
                    (label,)
                )
                conn.commit()
            finally:
                cursor.close()
                conn.close()

            await safe_close()
            return

        elif is_live and not matched:
            await send_face_response(False, "ใบหน้าจริงแต่ไม่ตรงกับข้อมูล", confidence=real_confidence, spoof_confidence=spoof_confidence)
        else:
            await send_face_response(False, "ยืนยันตัวตนไม่สำเร็จ", confidence=real_confidence, spoof_confidence=spoof_confidence)

    try:
        while True:
            if closed or websocket.client_state != WebSocketState.CONNECTED:
                break
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                logger_verify.info("Client disconnected")
                break
            except Exception as e:
                logger.exception(f"Error receiving data: {e}")
                await send_face_response(False, "Invalid input format")
                continue

            try:
                result, face, error = process_face_image(data, detector, sfas)
                if not result:
                    await send_face_response(False, error)
                    continue

                frame_rgb = cv2.cvtColor(result["frame"].copy(), cv2.COLOR_BGR2RGB)
                results_face = face_mesh.process(frame_rgb)
                results_hands = hands.process(frame_rgb)
            except Exception as e:
                logger.exception(f"processing failed: {e}")
                await send_face_response(False, "Face processing failed")
                continue

            is_occluded_by_hands = get_face_and_hand_data(result["frame"].copy(), results_face, results_hands)
            face_frontal = is_face_frontal(result["frame"].copy(), results_face)
            occluded = occlusion_detector.predict_occlusion(result["frame"])
            
            embedding = result.get("embedding")
            is_live = result.get("is_live")
            real_confidence = result.get("real_confidence")
            if real_confidence is None:
                real_confidence = 0.0
            spoof_confidence = result.get("spoof_confidence")
            
            try:
                matched, label, dist = compare_embedding(embedding, current_token_id)
            except Exception as e:
                logger.exception(f"compare_embedding failed: {e}")
                await send_face_response(False, "Comparison failed")
                continue

            await log_face_result(matched, label, dist, is_live, real_confidence, spoof_confidence, occluded, face, is_occluded_by_hands, face_frontal)

    except Exception as e:
        logger.exception(f"Unexpected error in websocket_verify: {e}")
    finally:
        await safe_close()

@app.websocket("/api/ws/face-register/")
async def websocket_register(websocket: WebSocket):
    global known_embeddings, known_labels
    await websocket.accept()
    embedding_saved = False
    confidence_scores = []
    collected_embeddings = []

    async def send_register_response(success, message, user_id=None, subuser_id=None, request_id=None, embedding=None, confidence=None,
                                     spoof_confidence=None, faces_detected=1, error_code=0):
        response = {
            "body": {
                "request_id": str(uuid.uuid4()),
                "user_id": user_id if user_id else None,
                "subuser_id": subuser_id if subuser_id else None,
                "success": success,
                "message": message,
                "confidence": round(float(confidence), 2) if confidence is not None else None,
                "spoof_confidence": round(float(spoof_confidence), 2) if spoof_confidence is not None else None,
                "faces_detected": faces_detected,
                "timestamp": format_thai_datetime(datetime.now(pytz.timezone("Asia/Bangkok"))),
                "error_code": error_code
            }
        }
        try:
            await websocket.send_json(response)
            logger_verify.info(json.dumps(response, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to send JSON: {e}")

    try:
        init_msg = await websocket.receive_text()
        init_data = json.loads(init_msg)
        token = init_data.get("token")
        request_id = init_data.get("request_id")

        if not token:
            await websocket.close(code=1008)
            return

        user = get_user_by_token(token)
        if not user:
            await send_register_response(False, "ไม่พบผู้ใช้", error_code=1001)
            await websocket.close(code=1008)
            return

        user_id = user["user_id"]
        
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "SELECT token_id FROM tokens WHERE api_token = %s",
            (token,)
        )
        token_row = c.fetchone()
        current_token_id = token_row[0] if token_row else None
        c.close()
        conn.close()

        if user["credit_count"] <= 0:
            await send_register_response(False, "โทเค็นที่คุณใช้ไม่ถูกต้องหรือหมดการใช้งาน", error_code=1011)
            await websocket.close(code=1008)
            return

        if int(user["activated"]) == 0:
            await send_register_response(False, "ยังไม่ได้ยืนยันบัญชี", error_code=1010)
            await websocket.close(code=1008)
            return

        username = user["username"]
        
    except Exception as e:
        logger.error(f"❌ Token validation failed: {e}")
        await websocket.close(code=1008)
        return
    
    try:
        while True:
            if embedding_saved:
                await websocket.close()
                break

            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                result, face, error = process_face_image(data, detector, sfas)
            except Exception:
                await send_register_response(False, "Face processing failed", error_code=5000)
                collected_embeddings.clear()
                confidence_scores.clear()
                continue

            if error:
                await send_register_response(False, error, error_code=4004)
                collected_embeddings.clear()
                confidence_scores.clear()
                continue
                
            frame_rgb = cv2.cvtColor(result["frame"].copy(), cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(frame_rgb)
            results_hands = hands.process(frame_rgb)

            embedding = result["embedding"]
            is_live = result["is_live"]
            real_confidence = result["real_confidence"]
            spoof_confidence = result["spoof_confidence"]
            
            face_frontal = is_face_frontal(result["frame"].copy(), results_face)
            occluded = occlusion_detector.predict_occlusion(result["frame"])
            is_occluded_by_hands = get_face_and_hand_data(result["frame"].copy(), results_face, results_hands)

            if not face_frontal:
                await send_register_response(False, "กรุณาหันใบหน้าตรง", confidence=real_confidence,
                                             spoof_confidence=spoof_confidence, error_code=1005)
                collected_embeddings.clear()
                confidence_scores.clear()
                continue

            if not is_live:
                await send_register_response(False, "ตรวจพบการปลอมใบหน้า", confidence=real_confidence,
                                             spoof_confidence=spoof_confidence, error_code=1001)
                collected_embeddings.clear()
                confidence_scores.clear()
                continue

            if face is None or occluded or is_occluded_by_hands:
                reason = "กรุณาเอามือออกจากใบหน้า" if is_occluded_by_hands else "กรุณาเอาสิ่งของออกจากใบหน้า"
                await send_register_response(False, reason, confidence=real_confidence,
                                             spoof_confidence=spoof_confidence, error_code=1004)
                collected_embeddings.clear()
                confidence_scores.clear()
                continue

            if real_confidence is not None:
                confidence_scores.append(real_confidence)
                if len(confidence_scores) > 5:
                    confidence_scores.pop(0)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            if avg_confidence >= 0.85:
                current_embeddings, current_labels, known_token_ids_from_db = load_known_embeddings_from_db()

                token_embeddings = [
                    emb for emb, tid in zip(current_embeddings, known_token_ids_from_db) 
                    if tid == current_token_id
                ]

                final_embedding_temp = np.mean(collected_embeddings + [embedding], axis=0) if collected_embeddings else embedding

                if token_embeddings:
                    distances = np.linalg.norm(np.array(token_embeddings) - final_embedding_temp, axis=1)
                    if np.min(distances) < 0.8:
                        await send_register_response(
                            False,
                            "ใบหน้านี้ลงทะเบียนไปแล้วภายใต้โทเค็นนี้",
                            confidence=avg_confidence,
                            spoof_confidence=spoof_confidence,
                            error_code=1006
                        )
                        collected_embeddings.clear()
                        confidence_scores.clear()
                        continue
                
                collected_embeddings.append(embedding)

                if len(collected_embeddings) < 5:
                    await send_register_response(
                        False,
                        f"กำลังเก็บข้อมูลหน้า... {len(collected_embeddings)}/5",
                        confidence=avg_confidence,
                        spoof_confidence=spoof_confidence,
                        error_code=1003
                    )
                    continue

                final_embedding = np.mean(collected_embeddings, axis=0)

                try:
                    conn = get_db_connection()
                    c = conn.cursor()

                    token_id_to_save = current_token_id

                    embedding_json = json.dumps(final_embedding.tolist())
                    c.execute(
                        "INSERT INTO subusers (user_id, token_id, face_embedding, registered_at, last_verified_at) "
                        "VALUES (%s, %s, %s, NOW(), NULL)",
                        (user_id, token_id_to_save, embedding_json)
                    )
                    subuser_id = c.lastrowid
                    
                    c.execute(
                        """
                        UPDATE users u
                        JOIN tokens t ON u.user_id = t.user_id
                        SET u.credit_count = u.credit_count - 1
                        WHERE t.api_token = %s
                        """,
                        (token,)
                    )

                    conn.commit()

                except Exception as e:
                    logger.error(f"Failed to save embedding to subusers: {e}")
                finally:
                    c.close()
                    conn.close()

                known_embeddings, known_labels, known_user_ids = load_known_embeddings_from_db()

                await send_register_response(True, "ลงทะเบียนใบหน้าสำเร็จ",
                                             user_id=user_id,
                                             subuser_id=subuser_id,
                                             embedding=final_embedding,
                                             confidence=avg_confidence,
                                             spoof_confidence=spoof_confidence,
                                             error_code=0)
                embedding_saved = True
            else:
                await send_register_response(False, "ตรวจพบการปลอมหน้า หรือความมั่นใจในการตรวจต่ำเกินไป",
                                             confidence=avg_confidence,
                                             spoof_confidence=spoof_confidence,
                                             error_code=1002)
                collected_embeddings.clear()
                confidence_scores.clear()

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")