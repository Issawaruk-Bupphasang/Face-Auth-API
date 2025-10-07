# occlusion_detector.py

import cv2
import torch
from torchvision import transforms
from PIL import Image
import yaml

from .model import Model
from .utils import load_weight

class OcclusionDetector:
    def __init__(self):
        self.configs = {
            "model_list": {
                "VGG16": "face_occlusion/weights/last_vgg16.pth"
            },
            "data": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "size": [500, 500]
            }
        }

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cpu")

        model_name = "VGG16"
        weight_path = self.configs["model_list"][model_name]
        
        self.model = self._load_model(model_name, self.device, weight_path)
        self.classes = {0: "non-occluded", 1: "occluded"}
        self.occluded_threshold = 0.8

    def _load_model(self, name, device, weight):
        name = name.lower().replace('-', '_')
        face_model = Model(name, 2, is_train=False).to(device)
        face_model = load_weight(face_model, weight, show=False)
        face_model.eval()
        return face_model

    def _transform_data(self, img):
        transform = transforms.Compose([
            transforms.Resize(self.configs["data"]["size"]),
            transforms.ToTensor(),
            transforms.Normalize(self.configs["data"]["mean"], self.configs["data"]["std"])
        ])
        return transform(img)

    def predict_occlusion(self, image):
        height, width = image.shape[:2]
        min_face_size = 80
        if height < min_face_size or width < min_face_size:
            return False

        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        trans_img = self._transform_data(img).to(self.device)

        with torch.no_grad():
            output = self.model(trans_img.unsqueeze(0))
            output = torch.softmax(output, 1)
            prob, pred = torch.max(output, 1)

        pred_class = self.classes[pred.item()]
        prob_val = prob.item()

        is_occluded = (pred.item() == 1 and prob_val > self.occluded_threshold)
        return is_occluded
