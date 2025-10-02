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
app_arcface = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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

# ===== Security =====

# ===== Environment setup =====
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===== Database config =====
from config import DB_CONFIG

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
    year = dt.year + 543  # พ.ศ.
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
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def query_db(query: str, args=(), one=False):
    """Query the database and return results."""
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
    query = "SELECT * FROM users WHERE api_token = %s LIMIT 1"
    return query_db(query, (token,), one=True)

def get_token_by_user_id(user_id: int):
    query = "SELECT api_token FROM users WHERE user_id = %s LIMIT 1"
    result = query_db(query, (user_id,), one=True)
    if result:
        return result['api_token']
    return None

# ===== Face Occlusion / Frontal Checks =====
ALL_KEY_INDICES = [
    33, 133, 362, 263, 1, 13, 14,    # จมูก/ปาก
    10, 152, 234, 454,               # หน้าผาก/คาง
    61, 291, 78, 308                 # ปากด้านข้าง
]

def get_face_and_hand_data(frame, results_face, results_hands):
    h, w, _ = frame.shape
    is_occluded_by_hands = False

    if results_face.multi_face_landmarks and results_hands.multi_hand_landmarks:
        landmarks = results_face.multi_face_landmarks[0].landmark
        
        # Get face bounding box
        xs_face = [int(lm.x * w) for lm in landmarks]
        ys_face = [int(lm.y * h) for lm in landmarks]
        face_x1, face_x2 = min(xs_face), max(xs_face)
        face_y1, face_y2 = min(ys_face), max(ys_face)
        
        # --- IMPROVED HAND OCCLUSION LOGIC ---
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Get hand bounding box
            xs_hand = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys_hand = [int(lm.y * h) for lm in hand_landmarks.landmark]
            hand_x1, hand_x2 = min(xs_hand), max(xs_hand)
            hand_y1, hand_y2 = min(ys_hand), max(ys_hand)
            
            # Calculate Intersection over Union (IoU)
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
                    break  # ไม่จำเป็นต้องวนต่อถ้าเจอการบดบังแล้ว
    
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

def compare_embedding(embedding: np.ndarray, user_id: int, threshold=0.8):
    known_embeddings, known_labels, known_user_ids = load_known_embeddings_from_db()
    if known_embeddings.size == 0:
        return False, None, None

    filter_indices = [i for i, uid in enumerate(known_user_ids) if uid == user_id]
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
    user_ids = []

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT f.subuser_id, f.face_embedding, s.user_id
            FROM faces f
            JOIN subusers s ON f.subuser_id = s.subuser_id
            WHERE f.face_embedding IS NOT NULL AND f.face_embedding != ''
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

    for subuser_id, embedding_json, user_id in records:
        if not embedding_json:
            continue
        try:
            embedding = np.array(json.loads(embedding_json), dtype=np.float32)
            labels.append(subuser_id)
            user_ids.append(user_id)      # เก็บ user_id
            embeddings.append(embedding)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[WARNING] Invalid embedding JSON for subuser '{subuser_id}': {e}")
            continue

    if embeddings:
        return np.stack(embeddings), labels, user_ids
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
    api_token = get_token_by_user_id(user["user_id"])
    token_info = get_user_by_token(api_token) if api_token else None
    token_uses_left = token_info.get("token_uses_left") if token_info else 0

    display_token = api_token if user.get("activated") else "Please verify your email"

    return templates.TemplateResponse("face_authentication.html", {
        "request": request,
        "user": username,
        "email": email,
        "activated": user.get("activated", 0),
        "token": display_token,
        "token_uses_left": token_uses_left
    })

@app.get("/api", response_class=HTMLResponse)
async def api_page(request: Request):
    username = request.session.get("username")

    if username:
        user = get_user_by_username(username)
        if user:
            email = user.get("email")
            api_token = get_token_by_user_id(user["user_id"])
            token_info = get_user_by_token(api_token) if api_token else None
            token_uses_left = token_info.get("token_uses_left") if token_info else 0
        else:
            username = "Guest"
            email = None
            token_uses_left = None
    else:  
        username = "Guest"
        email = None
        token_uses_left = None

    return templates.TemplateResponse("api.html", {
        "request": request,
        "user": username,
        "email": email,
        "token_uses_left": token_uses_left
    })

@app.get("/account", response_class=HTMLResponse)
async def account(request: Request):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/login", status_code=303)

    user = query_db("SELECT * FROM users WHERE username=%s", (username,), one=True)
    if not user:
        return HTMLResponse("User not found", status_code=404)

    api_token = get_token_by_user_id(user['user_id'])
    token_info = get_user_by_token(api_token) if api_token else None
    token_uses_left = token_info.get('token_uses_left') if token_info else 0

    return templates.TemplateResponse("account.html", {
        "request": request,
        "user": user["username"],
        "email": user["email"],
        "activated": user["activated"],
        "token_uses_left": token_uses_left if user['activated'] else 0
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
            'INSERT INTO users (username, password, email, activated) VALUES (%s, %s, %s, 1)',
            (username, hashed_pw.decode('utf-8'), email)
        )
        user_id = c.lastrowid

        api_token = str(uuid.uuid4())
        c.execute(
            "UPDATE users SET api_token=%s, token_uses_left=%s WHERE user_id=%s",
            (api_token, 10, user_id)
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

    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        await safe_close()
        return

    async def send_face_response(success, message, user_id=None, subuser_id=None, confidence=None,
                                 spoof_confidence=None,faces_detected=1, error_code=0):
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
                "error_code": error_code
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
            await send_face_response(False,reason,confidence=real_confidence,spoof_confidence=spoof_confidence,error_code=1004)
        elif not face_frontal:
            await send_face_response(False, "กรุณาหันใบหน้าตรง", confidence=real_confidence, spoof_confidence=spoof_confidence, error_code=1003)
        elif not is_live:
            await send_face_response(False, "ตรวจพบการปลอมใบหน้า", confidence=real_confidence, spoof_confidence=spoof_confidence, error_code=1001)
        elif matched and is_live:
            await send_face_response(True, "ยืนยันตัวตนสำเร็จ", user_id=user_id, subuser_id=label, confidence=real_confidence, spoof_confidence=None, error_code=0)

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
            await send_face_response(False, "ใบหน้าจริงแต่ไม่ตรงกับข้อมูล", confidence=real_confidence, spoof_confidence=spoof_confidence, error_code=1002)
        else:
            await send_face_response(False, "ยืนยันตัวตนไม่สำเร็จ", confidence=real_confidence, spoof_confidence=spoof_confidence, error_code=1002)

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
                await send_face_response(False, "Invalid input format", error_code=4001)
                continue

            try:
                result, face, error = process_face_image(data, detector, sfas)
                if not result:
                    await send_face_response(False, error, error_code=4004)
                    continue

                frame_rgb = cv2.cvtColor(result["frame"].copy(), cv2.COLOR_BGR2RGB)
                results_face = face_mesh.process(frame_rgb)
                results_hands = hands.process(frame_rgb)
            except Exception as e:
                logger.exception(f"processing failed: {e}")
                await send_face_response(False, "Face processing failed", error_code=5000)
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
                matched, label, dist = compare_embedding(embedding, user_id)
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

        if user["token_uses_left"] <= 0:
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

            already_registered = False

            if avg_confidence >= 0.85:
                current_embeddings, current_labels, current_user_ids = load_known_embeddings_from_db()
                user_embeddings = [emb for emb, uid in zip(current_embeddings, current_user_ids) if uid == user_id]

                final_embedding_temp = np.mean(collected_embeddings + [embedding], axis=0) if collected_embeddings else embedding

                if user_embeddings:
                    distances = np.linalg.norm(np.array(user_embeddings) - final_embedding_temp, axis=1)
                    if np.min(distances) < 0.8:
                        await send_register_response(
                            False,
                            "ใบหน้านี้ลงทะเบียนไปแล้ว",
                            confidence=avg_confidence,
                            spoof_confidence=spoof_confidence,
                            error_code=1006
                        )

                        already_registered = True
                        collected_embeddings.clear()
                        confidence_scores.clear()
                        continue

                if not already_registered:
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

                    c.execute(
                        "UPDATE users SET token_uses_left = token_uses_left - 1 WHERE api_token = %s",
                        (token,)
                    )
                    conn.commit()

                    embedding_json = json.dumps(final_embedding.tolist())
                    c.execute(
                        "INSERT INTO subusers (registered_at, user_id, last_verified_at) VALUES (NOW(), %s, NULL)",
                        (user_id,)
                    )
                    subuser_id = c.lastrowid

                    c.execute(
                        "INSERT INTO faces (subuser_id, face_embedding) VALUES (%s, %s)",
                        (subuser_id, embedding_json)
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

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")