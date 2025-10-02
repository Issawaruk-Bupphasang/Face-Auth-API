# anti_spoof_api.py

import os
import numpy as np
from face.src.anti_spoof_predict import AntiSpoofPredict
from face.src.generate_patches import CropImage
from face.src.utility import parse_model_name

class SFASPredictor:
    def __init__(self, model_dir="./face/resources/anti_spoof_models", device_id=0):
        self.model_dir = model_dir
        self.model_test = AntiSpoofPredict(device_id)
        self.image_cropper = CropImage()

    def predict_live(self, image):
        image_bbox = self.model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(self.model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False

            img = self.image_cropper.crop(**param)
            prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))

        sum_pred = np.sum(prediction)
        if sum_pred > 0:
            prediction = prediction / sum_pred

        # label = np.argmax(prediction)
        real_confidence = prediction[0][1]  # ความมั่นใจว่าเป็นของจริง
        spoof_confidence = prediction[0][0]  # ความมั่นใจว่าเป็นของปลอม
        is_live = real_confidence > spoof_confidence 
        # confidence = prediction[0][label] / 2
        # is_real = (label == 1)
        return is_live, real_confidence, spoof_confidence