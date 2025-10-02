# import cv2
# import mediapipe as mp

# # เตรียม mediapipe face mesh
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # โหลดภาพจากไฟล์
# image = cv2.imread("face.jpg")

# if image is None:
#     print("❌ ไม่พบไฟล์ input.jpg")
#     exit()

# # แปลง BGR -> RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# with mp_face_mesh.FaceMesh(
#     static_image_mode=True,   # โหมดภาพนิ่ง
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5
# ) as face_mesh:

#     # ประมวลผลหา face landmark
#     results = face_mesh.process(image_rgb)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             # วาดจุด + เส้นเชื่อม
#             mp_drawing.draw_landmarks(
#                 image=image,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
#             )
#             mp_drawing.draw_landmarks(
#                 image=image,
#                 landmark_list=face_landmarks,
#                 connections=mp_face_mesh.FACEMESH_CONTOURS,
#                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,0), thickness=1, circle_radius=1),
#                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
#             )

#     # บันทึกผลลัพธ์
#     cv2.imwrite("output.jpg", image)
#     print("✅ บันทึกผลลัพธ์เป็น output.jpg แล้ว")

#     # แสดงผล
#     cv2.imshow("Face Mesh Result", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

import onnxruntime as ort

print("Available providers:", ort.get_available_providers())
