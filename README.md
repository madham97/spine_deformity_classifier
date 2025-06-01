# Lumbar Spine Degenerative Classification

This project focuses on developing a deep learning model to identify and localize key anatomical features in spinal MRI images, aiding in the detection of degenerative conditions. Detailed report attached in Documentation.pdf

## Project Overview

The goal is to classify lumbar spine degenerative conditions using a deep learning model that combines classification with spatial offset prediction, enhancing diagnostic accuracy in spinal MRI scans.

## Key Components

### Dataset

- **RSNA Lumbar Spine Dataset:** Curated for analyzing degenerative lumbar spine conditions via MRI imaging, including severity labels and spatial coordinates for precise localization.

### Imaging and Anatomical Overview

- **MRI Planes:** Axial, sagittal, and coronal views provide comprehensive insights into spinal structures.
- **Spinal Regions:** Cervical, thoracic, lumbar, and sacral regions each have unique characteristics relevant to degeneration analysis.

### Model Architecture

- **DeepLabV3+ with ResNet Backbone:** Utilized for pixel-wise segmentation and classification maps.
- **Attention Modules:** Position Attention Module (PAM) and Channel Attention Module (CAM) enhance feature integration.

## Methodology

### Data Preprocessing

- Images resized to 512x512 resolution for uniformity.
- Heatmaps and offset maps generated to facilitate learning of key points in spinal images.

### Model Pipeline

- **Feature Extraction:** Modified ResNet-50 captures essential details for keypoint localization.
- **Heatmap and Offset Heads:** Output spatial probability maps and displacement vectors for refined localization.

### Loss Function

- Combines Binary Cross-Entropy (BCE) and Mean Absolute Error (MAE) to balance localization and classification objectives.

## Experiments

### Default Settings

- Initial experiments followed the SpineOne paper's methodology, achieving moderate success in differentiating severity classes.

### Ablation Studies

- Explored the impact of excluding offset loss and attention modules, revealing their importance in model performance.

## Conclusion

The model demonstrates potential for clinical utility in diagnosing spinal deformities. Future work could optimize architecture and integrate additional data modalities for improved accuracy.
