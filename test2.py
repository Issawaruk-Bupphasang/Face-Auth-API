import cv2
import mediapipe as mp
import numpy as np

# ===== Initialize MediaPipe FaceMesh =====
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== Initialize MediaPipe Hands =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# keypoints ครอบคลุมใบหน้าสำคัญ
ALL_KEY_INDICES = [
    33, 133, 362, 263, 1, 13, 14,       # จมูก/ปาก
    10, 152, 234, 454,                  # หน้าผาก/คาง
    61, 291, 78, 308                    # ปากด้านข้าง
]

def get_face_and_hand_data(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results_face = face_mesh.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    face_detected = False
    occlusion_score = 0.0
    hands_detected = False
    is_occluded_by_hands = False

    if results_face.multi_face_landmarks:
        face_detected = True
        landmarks = results_face.multi_face_landmarks[0].landmark
        
        occlusion_score = 100.0
        
        if results_hands.multi_hand_landmarks:
            hands_detected = True
            
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
                    if iou > 0.15:  # Use a robust IoU threshold (e.g., 15%)
                        is_occluded_by_hands = True
                        occlusion_score -= min(iou * 100, 100) # Reduce score based on overlap percentage
            
        # Check missing keypoints
        visible_points = 0
        for idx in ALL_KEY_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                if 0 <= px < w and 0 <= py < h:
                    visible_points += 1
        
        keypoints_ratio = visible_points / len(ALL_KEY_INDICES)
        occlusion_score -= (1.0 - keypoints_ratio) * 100
        
        # Check texture
        if keypoints_ratio > 0.5:
            nose_tip = landmarks[1]
            cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
            r = 40
            x1, y1 = max(0, cx-r), max(0, cy-r)
            x2, y2 = min(w, cx+r), min(h, cy+r)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if np.std(gray) <= 10:
                    occlusion_score -= 20
    
    return face_detected, is_occluded_by_hands, hands_detected, max(0.0, occlusion_score)

def is_face_frontal(frame, yaw_thresh=15, pitch_min=0.3, pitch_max=0.5):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return False

    lm = results.multi_face_landmarks[0].landmark
    left_eye = lm[33]
    right_eye = lm[263]
    nose = lm[1]
    chin = lm[152]

    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
    nx, ny = int(nose.x * w), int(nose.y * h)
    cx, cy = int(chin.x * w), int(chin.y * h)

    eye_mid_x = (lx + rx) / 2
    nose_offset = nx - eye_mid_x
    yaw_ok = abs(nose_offset) < yaw_thresh

    eye_mid_y = (ly + ry) / 2
    vertical_ratio = (ny - eye_mid_y) / (cy - eye_mid_y)
    pitch_ok = pitch_min < vertical_ratio < pitch_max
    
    return yaw_ok and pitch_ok

# ===== main loop =====
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้")
        return

    print("✅ เปิดกล้องสำเร็จ (กด ESC เพื่อออก)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get all necessary data
        face_detected, is_occluded_by_hands, hands_detected, occlusion_score = get_face_and_hand_data(frame.copy())
        is_frontal = is_face_frontal(frame.copy())
        
        y_pos = 50  # Starting Y position for text
        
        # Line 1: Face Detection Status
        if face_detected:
            text1 = "Face: Detected"
            color1 = (0, 255, 0)
        else:
            text1 = "Face: Not Detected"
            color1 = (0, 0, 255)
        cv2.putText(frame, text1, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color1, 3)
        
        # Line 2: Occlusion Score
        y_pos += 50
        text2 = f"Occlusion Score: {occlusion_score:.1f}%"
        if occlusion_score < 70:
            color2 = (0, 165, 255) # Orange
        else:
            color2 = (0, 255, 0) # Green
        cv2.putText(frame, text2, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color2, 3)

        # Line 3: Frontal Status
        y_pos += 50
        if is_frontal:
            text3 = "Frontal: YES"
            color3 = (0, 255, 0)
        else:
            text3 = "Frontal: NO"
            color3 = (0, 255, 255)
        cv2.putText(frame, text3, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color3, 3)

        # Line 4: Hand Presence Status
        y_pos += 50
        if hands_detected:
            text4 = "Hands: Detected"
            color4 = (0, 0, 255)
        else:
            text4 = "Hands: Not Detected"
            color4 = (0, 255, 0)
        cv2.putText(frame, text4, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color4, 3)

        cv2.imshow("Multi-Status Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()