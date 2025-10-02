# from io import BytesIO
# import yaml

# from PIL import Image
# import streamlit as st
# import torch
# from torchvision import transforms

# from model import Model
# from utils import load_weight


# # Configs
# with open("configs/demo_configs.yml") as f:
#     confis = yaml.load(f, Loader=yaml.SafeLoader)

# CLASSES = {0: "non-occluded",
#            1: "occluded"}


# def transform_data(img):
#     transform = transforms.Compose([
#             transforms.Resize(confis["data"]["size"]),
#             transforms.ToTensor(),
#             transforms.Normalize(confis["data"]["mean"], confis["data"]["std"])
#     ])
#     return transform(img)


# def model(name, device, weight):
#     name = name.lower()
#     name = name.replace('-', '_')
#     faceocc_model = Model(name, 2, is_train=False).to(device)
#     faceocc_model = load_weight(faceocc_model, weight, show=False)
#     faceocc_model.eval()
#     return faceocc_model


# # def convert_pil(img):
# #     pil_image = Image.open(BytesIO(img.getvalue()))
# #     return pil_image


# def convert_pil(img):
#     pil_image = Image.open(BytesIO(img.getvalue())).convert("RGB")
#     return pil_image

# def classification(name, img, device, weight):
#     img = convert_pil(img)
#     trans_img = transform_data(img).to(device)
#     output = model(name, device, weight)(trans_img.unsqueeze(0))
#     output = torch.softmax(output, 1)
#     prob, pred = torch.max(output, 1)
#     return prob, pred


# def main():
#     st.set_page_config(
#         page_title="Face occlusion demo",
#         layout="wide"
#     )
#     st.title("Face occlusion demo")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         select_model = col1.selectbox("Choose model", confis["model_list"].keys())
#         up_img = col1.file_uploader("Upload face image")
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     with col2:
#         if up_img:
#             st.image(up_img, width=300)
#             pro, pre = classification(select_model, up_img, device, confis["model_list"][select_model])

#     with col3:
#         if up_img:
#             with st.spinner(text="Wait for it..."):
#                 st.write(
#                     {
#                         "model": select_model,
#                         "prob": f"{round(pro.item() * 100, 2)} %",
#                         "label": CLASSES[pre.item()]
#                     }
#                 )


# if __name__ == "__main__":
#     main()

import cv2
import torch
from torchvision import transforms
from PIL import Image

from model import Model
from utils import load_weight
import yaml

# โหลด config
with open("configs/demo_configs.yml") as f:
    confis = yaml.load(f, Loader=yaml.SafeLoader)

CLASSES = {0: "non-occluded", 1: "occluded"}

# ฟังก์ชัน transform
def transform_data(img):
    transform = transforms.Compose([
        transforms.Resize(confis["data"]["size"]),
        transforms.ToTensor(),
        transforms.Normalize(confis["data"]["mean"], confis["data"]["std"])
    ])
    return transform(img)

# สร้างโมเดล
def load_model(name, device, weight):
    name = name.lower().replace('-', '_')
    face_model = Model(name, 2, is_train=False).to(device)
    face_model = load_weight(face_model, weight, show=False)
    face_model.eval()
    return face_model

# ฟังก์ชันทำนาย
def classify_image(model, img, device):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    trans_img = transform_data(img).to(device)
    with torch.no_grad():
        output = model(trans_img.unsqueeze(0))
        output = torch.softmax(output, 1)
        prob, pred = torch.max(output, 1)
    return prob.item(), pred.item()

# เลือก device
device = "cuda" if torch.cuda.is_available() else "cpu"

# โหลดโมเดล
model_name = list(confis["model_list"].keys())[0]  # เลือกโมเดลตัวแรก
weight_path = confis["model_list"][model_name]
model = load_model(model_name, device, weight_path)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    prob, pred = classify_image(model, frame, device)
    
    # แสดงผลบนภาพ
    label = f"{CLASSES[pred]}: {prob*100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Face Occlusion Detection", frame)
    
    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
