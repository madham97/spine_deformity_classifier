
import os
import pandas as pd
import numpy as np
import pydicom
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class SpineDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None, roi_size=64):
        """
        Args:
            annotations (pd.DataFrame): DataFrame containing image paths and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            roi_size (int): Size of the square ROI to extract.
        """
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform
        self.roi_size = roi_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx]['image_path'])
        label = self.annotations.iloc[idx]['severity']

        dicom_image = pydicom.dcmread(img_path)
        image = dicom_image.pixel_array

        # Normalize 
        image = image.astype(np.float32)
        image -= np.min(image)
        image /= np.max(image)
        image *= 255.0

        image = Image.fromarray(image).convert('L')

        # Extract ROI
        x_center = self.annotations.iloc[idx]['x']
        y_center = self.annotations.iloc[idx]['y']
        roi = self.extract_roi(image, x_center, y_center)

        if self.transform:
            roi = self.transform(roi)

        return roi, label


#also wanna try and use a gaussian heatmap instead of a square ROI.
    def extract_roi(self, image, x_center, y_center):
        """
        Extracts a square ROI from the image centered at (x_center, y_center).
        """
        x_center = int(x_center)
        y_center = int(y_center)
        half_size = self.roi_size // 2

        left = max(x_center - half_size, 0)
        upper = max(y_center - half_size, 0)
        right = left + self.roi_size
        lower = upper + self.roi_size

        image_width, image_height = image.size
        if right > image_width:
            right = image_width
            left = right - self.roi_size
        if lower > image_height:
            lower = image_height
            upper = lower - self.roi_size

        roi = image.crop((left, upper, right, lower))
        return roi

def load_annotations(train_csv, coords_csv, conditions, levels):
    """
    Merges labels and coordinates to create a DataFrame for training.
    """
    import os
    import pandas as pd

    labels_df = pd.read_csv(train_csv)
    coords_df = pd.read_csv(coords_csv)

    label_columns = [f"{condition}_{level}" for condition in conditions for level in levels]
    labels_df = labels_df[['study_id'] + label_columns]

    labels_df = labels_df.melt(id_vars=['study_id'], var_name='condition_level', value_name='severity')

    split_cols = labels_df['condition_level'].str.rsplit('_', n=2, expand=True)
    labels_df['condition'] = split_cols[0]
    labels_df['level'] = split_cols[1] + '_' + split_cols[2]
    labels_df.drop('condition_level', axis=1, inplace=True)

    labels_df['condition'] = labels_df['condition'].str.lower()
    labels_df['level'] = labels_df['level'].str.lower()
    coords_df['condition'] = coords_df['condition'].str.lower()
    coords_df['level'] = coords_df['level'].str.lower()

    # Merge 
    merged_df = pd.merge(labels_df, coords_df, on=['study_id', 'condition', 'level'])

    #resulting df should have columns: study_id,series_id,instance_number, condition, level, severity, x, y.

    # Map severity to numerical labels
    severity_mapping = {'normal': 0, 'mild': 0, 'normal/mild': 0, 'moderate': 1, 'severe': 2}
    merged_df['severity'] = merged_df['severity'].str.lower().map(severity_mapping)

    merged_df = merged_df.dropna(subset=['severity'])

    merged_df['image_path'] = merged_df.apply(
        lambda row: os.path.join(
            str(row['study_id']),
            str(row['series_id']),
            f"{row['instance_number']}.dcm"
        ), axis=1)

    merged_df['severity'] = merged_df['severity'].astype(int)
    #columns: study_id,series_id,instance_number, condition, level, severity, x, y, image_path
    return merged_df


def get_transforms():
    """
    Returns data augmentation transforms.
    """
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return data_transforms
