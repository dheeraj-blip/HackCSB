import pytesseract
from PIL import Image

def EXTRACT(img):

    # Specify the OCR engine mode and page segmentation mode
    custom_config = r'--oem 1 --psm 6'

    # Perform OCR on the image
    text = pytesseract.image_to_string(img, config=custom_config)

    return text