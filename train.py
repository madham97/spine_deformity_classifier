
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from data_preprocess import SpineDataset, get_transforms, load_annotations
from model import SpineClassifier
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    best_loss = np.inf

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Compute epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy 
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_model.pth')

    print('Training complete')
    print(f'Best validation loss: {best_loss:.4f}')

def main():
    # Load annotations
    conditions = ['left_neural_foraminal_narrowing', 'right_neural_foraminal_narrowing',
                  'left_subarticular_stenosis', 'right_subarticular_stenosis',
                  'spinal_canal_stenosis']
    levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
    annotations = load_annotations('train.csv', 'train_label_coordinates.csv', conditions, levels)

    # Split into training and validation sets
    train_df, val_df = train_test_split(annotations, test_size=0.2, random_state=42, stratify=annotations['severity'])

    # Data transformations
    data_transforms = get_transforms()

    # Create datasets
    train_dataset = SpineDataset(train_df, root_dir='train_images', transform=data_transforms)
    val_dataset = SpineDataset(val_df, root_dir='train_images', transform=data_transforms)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize model
    model = SpineClassifier(num_classes=3, pretrained=True)

    # Define criterion with sample weights
    class_weights = torch.tensor([1.0, 2.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10)

if __name__ == '__main__':
    main()
