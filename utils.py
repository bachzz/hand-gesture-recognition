from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np

def pil_image_to_base64(cv2_image):
    color_coverted = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())

def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def base64_to_pil_image(base64_img):
    return toRGB(Image.open(BytesIO(base64.b64decode(base64_img))))
