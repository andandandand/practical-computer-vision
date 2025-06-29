#!/usr/bin/env python
# coding: utf-8

# # Starter Notebook: Fruit Classification Challenge
# 
# This notebook provides the train, testing, and validation data loaders.
# 
# Through the following link, you can access a private Kaggle challenge on fruit classification:
# 
# https://www.kaggle.com/t/b4dbb9add11c4da0962b837929799d52
# 
# ## Challenge
# Build a model to classify images of bananas, apples, and oranges as fresh or rotten.
# Dataset
# 
# * Training set: 1,058 labeled images across fruit types
# * Test set: 453 images (hidden labels)
# * Resolution: varied, RGBA format
# 
# ## Build a Model and a Training/Validation Loop
# 
# You can use the following code as a starting point for the challenge. See if you can achieve over 92% accuracy. 
# 

# In[1]:


import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define constants
TRAIN_CSV_PATH = '/kaggle/input/fruit-classification/train.csv'
TRAIN_IMG_DIR = '/kaggle/input/fruit-classification/train/train'
TEST_IMG_DIR = '/kaggle/input/fruit-classification/test/test'
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define the transformation pipeline
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class for the training and validation datasets (with labels)
class FruitDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        # Create a label mapping for classification
        self.classes = ['fresh_apple', 'fresh_banana', 'fresh_orange', 
                        'rotten_apple', 'rotten_banana', 'rotten_orange']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Id']
        label = self.df.iloc[idx]['Label']
        img_path = os.path.join(self.img_dir, img_name)

        # Handle both PNG and JPEG files with RGBA
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Convert text label to numeric index
        label_idx = self.class_to_idx[label]

        return image, label_idx

# Class for the test dataset (without labels)
class TestFruitDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Handle both PNG and JPEG files with RGBA
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name  # Return image name as identifier instead of label

def create_dataloaders():
    # Read the training data CSV
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    # Split into train and validation sets
    train_df, val_df = train_test_split(
        train_df, test_size=VALIDATION_SPLIT, random_state=SEED, 
        stratify=train_df['Label']  # Stratify to maintain class distribution
    )

    # Create datasets
    train_dataset = FruitDataset(train_df, TRAIN_IMG_DIR, transform=train_transforms)
    val_dataset = FruitDataset(val_df, TRAIN_IMG_DIR, transform=val_test_transforms)
    test_dataset = TestFruitDataset(TEST_IMG_DIR, transform=val_test_transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader, train_dataset.classes

# Function to visualize a batch of images
def visualize_batch(dataloader, classes=None, n=8):
    images, labels = next(iter(dataloader))
    images = images[:n]  # Take only n images

    # Move to CPU and convert to numpy for visualization
    images = images.cpu().numpy()

    # Denormalize images for better visualization
    images = np.transpose(images, (0, 2, 3, 1))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)

    # Plot images in a grid
    fig, axes = plt.subplots(2, n//2, figsize=(15, 8))
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        if classes and isinstance(labels, torch.Tensor):
            ax.set_title(f"Label: {classes[labels[i].item()]}")
        elif not isinstance(labels, torch.Tensor):
            # For test datasets where labels are filenames
            ax.set_title(f"File: {labels[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# In[2]:


train_loader, val_loader, test_loader, classes = create_dataloaders()


# In[3]:


print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")
print(f"Classes: {classes}")


# In[4]:


# # Visualize a batch from the train_loader
visualize_batch(train_loader, classes)  # Training samples with augmentation


# In[5]:


visualize_batch(val_loader, classes)  # Validation samples, no augmentation


# In[6]:


visualize_batch(test_loader, classes)  # Test samples, no augmentation

