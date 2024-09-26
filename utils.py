
import pydicom
import numpy as np
from PIL import Image

def load_dicom_image(image_path):
    """
    Loads a DICOM image and returns it as a PIL Image.
    """
    dicom_image = pydicom.dcmread(image_path)
    image = dicom_image.pixel_array

    # Normalize image to [0, 255]
    image = image.astype(np.float32)
    image -= np.min(image)
    image /= np.max(image)
    image *= 255.0

    # Convert to PIL Image
    image = Image.fromarray(image).convert('L')
    return image
