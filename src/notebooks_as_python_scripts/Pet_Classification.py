#!/usr/bin/env python
# coding: utf-8

# ## Pet Breed Classification - Getting Started with Image Classification in  PyTorch and FiftyOne
# 
# #### Author: Antonio Rueda-Toicen
# 
# **antonio.rueda.toicen 'at' hpi 'dot' de**
# 
# 
# [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
# 
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

# Approaches for Image Classification: Fully Supervised vs Zero Shot
# 
# This notebook shows how to recognize different breeds of cats and dogs.
# 
# We'll use a collection of pet images and a fine-tuned PyTorch neural network (a [ResNet 18](https://arxiv.org/abs/1512.03385)) to identify them.
# 
# **Here's what we'll do:**
# 
# 1. **Get the images:** We'll use a dataset called "Oxford-IIIT Pets" which contains many pictures of cats and dogs.
# 2. **Prepare the images:** We'll make sure the images are in the right format for the neural network to process.
# 3. **Train the network:** We'll use a special type of program called a "neural network" to train the computer to recognize the pets.
# 4. **Test the network:** We'll see how well the computer can identify new pet images it hasn't seen before.
# 5. **Explore the results:** We'll use the open source FiftyOne app to visualize the predictions and understand deeper both the neural network and the image dataset.
# 6. **Compare the OOD results with those obtained from a CLIP model**: Zero-shot classification with vision-language foundation models (e.g. [CLIP](https://arxiv.org/abs/2103.00020)) is a powerful tool for image labeling. We will introduce the technique near the end of this notebook and go deeper into it in future sessions.
# 
# 

# ## The Oxford pets dataset
# 
# The [Oxford-IIIT pets dataset by O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
# ](https://www.robots.ox.ac.uk/~vgg/data/pets/) is a 37 category pet dataset with roughly 200 images for each class. Each class corresponds to a breed of cat or dog. The dataset includes annotations for classification, semantic segmentation and object detection. In this notebook, we will focus on image classification and use the [version of the dataset available on PyTorch datasets](https://pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html).

# In[ ]:


# These are the breeds of dogs and cats considered in the Oxford-IIT pets dataset
oxford_pets_classes = [
    ("Abyssinian", "cat"),
    ("American Bulldog", "dog"),
    ("American Pit Bull Terrier", "dog"),
    ("Basset Hound", "dog"),
    ("Beagle", "dog"),
    ("Bengal", "cat"),
    ("Birman", "cat"),
    ("Bombay", "cat"),
    ("Boxer", "dog"),
    ("British Shorthair", "cat"),
    ("Chihuahua", "dog"),
    ("Egyptian Mau", "cat"),
    ("English Cocker Spaniel", "dog"),
    ("English Setter", "dog"),
    ("German Shorthaired", "dog"),
    ("Great Pyrenees", "dog"),
    ("Havanese", "dog"),
    ("Japanese Chin", "dog"),
    ("Keeshond", "dog"),
    ("Leonberger", "dog"),
    ("Maine Coon", "cat"),
    ("Miniature Pinscher", "dog"),
    ("Newfoundland", "dog"),
    ("Persian", "cat"),
    ("Pomeranian", "dog"),
    ("Pug", "dog"),
    ("Ragdoll", "cat"),
    ("Russian Blue", "cat"),
    ("Saint Bernard", "dog"),
    ("Samoyed", "dog"),
    ("Scottish Terrier", "dog"),
    ("Shiba Inu", "dog"),
    ("Siamese", "cat"),
    ("Sphynx", "cat"),
    ("Staffordshire Bull Terrier", "dog"),
    ("Wheaten Terrier", "dog"),
    ("Yorkshire Terrier", "dog")
]


# ## Install fiftyone and wandb

# In[ ]:


# Install fiftyone, dacite 1.6 is a dependency for it
get_ipython().system('pip install dacite==1.6 -q')
get_ipython().system('pip install fiftyone==1.3.2 -q')
# Install wandb
get_ipython().system('pip install wandb==0.19 -q')


# In[ ]:


# Check fiftyone install
import fiftyone as fo
fo.__version__


# In[ ]:


# Check wandb install
import wandb
wandb.__version__


# ## Library imports

# In[ ]:


import os
import io
import random
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from google.colab import userdata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision import models
from torchvision.datasets import OxfordIIITPet
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor



# ## Get wandb API key from Google Colab

# In[ ]:


os.environ['WANDB_API_KEY'] = userdata.get('wandb_api')


# ## Define wandb project

# In[ ]:


project="pets-classification"
# https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-
wandb.init(settings=wandb.Settings(start_method="fork"), project=project)


# ## Seeding for reproducibility

# In[ ]:


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


# # Understanding Random Seed Setting in PyTorch
# 
# ## What the Code Does
# 
# ```python
# def set_seed(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# ```
# 
# This function makes our training experiment repeatable by controlling all the parts of our code that normally use randomness. Think of it like setting the starting point for all random choices our program will make.
# 
# | Component | Purpose |
# |-----------|---------|
# | `random.seed()` | Sets Python's built-in random module seed |
# | `PYTHONHASHSEED` | Controls hash randomization in Python |
# | `np.random.seed()` | Sets NumPy's random number generator seed |
# | `torch.manual_seed()` | Sets PyTorch's CPU-based random generators |
# | `torch.cuda.manual_seed()` | Sets PyTorch's GPU-based random generators |
# | `cudnn.deterministic = True` | Makes cuDNN operations deterministic |
# | `cudnn.benchmark = False` | Disables cuDNN auto-tuner optimization |
# 
# ## Why It Matters
# 
# In deep learning, randomness affects many processes:
# - Weight initialization
# - Batch sampling order
# - Data augmentation
# - Dropout behavior
# 
# Setting fixed seeds ensures:
# - Reproducible results across runs
# - Easier debugging
# - Fair model comparisons
# - Reliable research findings
# 
# The trade-off is potentially slightly lower performance, as some non-deterministic optimizations are disabled.
# 
# ### What non-deterministic operations are disabled?
# 
# When we set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`, we're disabling several non-deterministic optimizations that would otherwise improve performance. Here's what gets disabled:
# 
# 1. **cuDNN Auto-Tuner**: When `cudnn.benchmark = False`, PyTorch disables the cuDNN auto-tuner, which normally:
#    - Benchmarks multiple convolution algorithms at the beginning of training
#    - Dynamically selects the fastest algorithm for our specific hardware and input dimensions
#    - Adapts to different input sizes by finding optimal algorithms for each
# 
# 2. **Parallel Algorithm Selection**: cuDNN normally can choose from multiple implementations of the same operation based on runtime conditions, potentially using:
#    - Different memory access patterns
#    - Hardware-specific optimizations
#    - Concurrent execution strategies
# 
# 3. **Dynamic Memory Management**: Some memory optimizations that dynamically reallocate or reuse memory in ways that can vary between runs are constrained.
# 
# 4. **Asynchronous Execution Patterns**: Non-deterministic scheduling of operations across CPU/GPU that might complete in different orders between runs.
# 
# 5. **Sub-optimal Convolution Algorithms**: Instead of using the fastest algorithm, PyTorch will use deterministic algorithms which may be slower but guarantee the same results every time.
# 
# The performance impact varies widely depending on our model architecture and hardware, but typically ranges from negligible to around 10-15% slowdown in training time. For models with many convolution operations, the impact can be more significant.
# 
# For day-to-day experimentation, we might want to enable determinism only when debugging or for final result verification, then use the optimized settings for regular training.

# ## Check availability of GPU

# In[ ]:


# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ## Initialize wandb

# In[ ]:


# Initialize W&B
wandb.init(project="oxford-pets-classification",
           name="resnet18-transfer-learning",
           config={
               "architecture": "ResNet18",
               "dataset": "Oxford-IIIT Pet Dataset",
               "learning_rate": 0.001,
               "epochs": 10,
               "batch_size": 32,
               "optimizer": "Adam",
               "scheduler": "ReduceLROnPlateau"
           })


# ## Define data transformations

# In[ ]:


# Data Preparation
# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset
print("Loading Oxford-IIIT Pet Dataset...")
data_root = './data'
os.makedirs(data_root, exist_ok=True)

# Download the dataset
train_dataset = OxfordIIITPet(root=data_root, split='trainval', download=True, transform=train_transforms)
test_dataset = OxfordIIITPet(root=data_root, split='test', download=True, transform=val_test_transforms)

# Check sizes
len(train_dataset), len(test_dataset)


# In[ ]:


# Split train dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])


# ## Transformations for validation set

# In[ ]:


# Override transform for validation dataset
# It had been assigned the transformation assigned to the training set
val_dataset.dataset.transform = val_test_transforms


# ## Create DataLoaders

# In[ ]:


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")


# In[ ]:


# Display some sample images
def display_sample_images(dataset, num_samples=5):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        img, label = dataset[idx]
        img = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"{oxford_pets_classes[label][0]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

display_sample_images(train_dataset)


# ## Define a ResNet18 as the pretrained model

# In[ ]:


# Model definition
class PetClassifier(nn.Module):
    def __init__(self, num_classes=37):
        super(PetClassifier, self).__init__()
        # Load a pre-trained ResNet18 model
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')

        # Freeze the early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(num_features, 256),  # New hidden layer with 256 units
            nn.ReLU(),  # Activation function for the hidden layer
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.backbone(x)

# Initialize the model
model = PetClassifier(num_classes=37).to(device)
# Check the architecture of the model
model




# ## Define the loss function and variant of stochastic gradient descent

# In[ ]:


criterion = nn.CrossEntropyLoss()
initial_learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)


# ## Define a scheduler for the learning rate

# ### OneCycleLR: A Dynamic Learning Rate Scheduler
# 
# [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) is a learning rate scheduler that varies the learning rate during training, following a specific pattern:
# 
# 1. **Warm-up:** The learning rate starts low and gradually increases to a maximum value (`max_lr`).
# 2. **Annealing:** After reaching the maximum, the learning rate gradually decreases to a very small value.
# 
# **Benefits:**
# 
# - **Faster Convergence:**  Helps models train faster and reach good results in fewer epochs.
# - **Improved Generalization:** Can lead to better performance on unseen data by preventing overfitting.
# - **Reduced Tuning:** Often requires less hyperparameter tuning compared to other schedulers.
# 
# **Key Hyperparameter:** `max_lr` (the maximum learning rate) needs to be tuned for optimal performance.

# In[ ]:


num_epochs = 5
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=len(train_loader)*num_epochs)


# ## Training loop

# In[ ]:


# Training loop
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ## Evaluation

# In[ ]:


# Validation/Testing function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, cm, all_preds, all_labels


# ## Running the training loop

# In[ ]:


best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Train one epoch
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # Evaluate on validation set
    val_loss, val_acc, val_precision, val_recall, val_cm, _, _ = evaluate(model, val_loader, criterion, device)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "learning_rate": optimizer.param_groups[0]['lr']
    })


    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save locally
        torch.save(model.state_dict(), 'best_model.pth')
        # Save to wandb
        wandb.save('best_model.pth')
        print("Saved best model locally and to wandb!")


# In[ ]:


# After the training loop:
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')

# Find the epoch with the lowest validation loss
min_val_loss_epoch = np.argmin(val_losses)

# Highlight the minimum validation loss point
plt.plot(min_val_loss_epoch, val_losses[min_val_loss_epoch], 'ro', markersize=8,
         label='Minimum Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ## Test Set Evaluation

# In[ ]:


# Assuming you have the test_loader, model, and oxford_pets_classes

def display_batch_predictions(model, test_loader, oxford_pets_classes, batch_size=5):
    """Displays predictions for a batch of images from the test set."""

    # Get a batch of data
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)  # Move inputs to the device

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_classes = torch.max(outputs, 1)

    # Display predictions
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
    for i in range(batch_size):
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]) # Denormalize
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {oxford_pets_classes[predicted_classes[i].item()][0]}\n"
                          f"True: {oxford_pets_classes[labels[i].item()][0]}\n"
                          f"Prob: {probabilities[i][predicted_classes[i].item()].item():.4f}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to display predictions
display_batch_predictions(model, test_loader, oxford_pets_classes)


# In[ ]:


# Evaluate on test set
print("Evaluating on test set...")
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc, test_precision, test_recall, test_cm, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")

# Log final test results to W&B
wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_acc,
    "test_precision": test_precision,
    "test_recall": test_recall
})


# Get only the breed names from oxford_pets_classes
breed_labels = [breed for breed, _ in oxford_pets_classes]

# Display confusion matrix with breed labels
plt.figure(figsize=(12, 10))
plt.imshow(test_cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(np.arange(len(breed_labels)), breed_labels, rotation=90)  # Set breed labels for x-axis
plt.yticks(np.arange(len(breed_labels)), breed_labels)  # Set breed labels for y-axis
plt.tight_layout()
plt.show()


# In[ ]:


# Create a readable report
class_report = classification_report(test_labels, test_preds, output_dict=True)
class_metrics = pd.DataFrame(class_report).transpose()

# Extract breed names from oxford_pets_classes
breed_names = [breed for breed, _ in oxford_pets_classes]

# Add a new column 'breed' to the DataFrame
class_metrics['breed'] = breed_names + ['accuracy', 'macro avg', 'weighted avg']
# The last 3 elements of class_metrics.index are 'accuracy', 'macro avg', 'weighted avg'

# Set the 'breed' column as the index
class_metrics.set_index('breed', inplace=True)
class_metrics


# ## Why we need `wandb.finish()`
# 
# `wandb.finish()` tells Weights & Biases (`wandb`) that your experiment is done. This is important for a few reasons:
# 
# - **Saves your data:** Makes sure all your results are safely stored online.
# - **Frees up resources:**  `wandb` stops using computing power for your finished experiment.
# - **Organizes experiments:** Helps you keep track of different runs and compare results.
# 
# If you use `wandb.log()` after calling `wandb.finish()`, you'll get an error. You need to call `wandb.init()` again to start a new experiment before logging more data.

# In[ ]:


wandb.finish()


# In[ ]:


# Example of inference
def predict_image(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        output = model(image_tensor)
         # Get all probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
    # Get probability of predicted class
    predicted_class_probability = probabilities[0][predicted.item()]

    return predicted.item(), predicted_class_probability.item()


# ## Testing your own images

# Try testing with "Out of Distribution" images, with true labels different than what we have in the training set. For example, we believe our dog Rita was a German Shepherd mix.

# In[ ]:


img_url = 'https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/refs/heads/main/rita_cropped.jpg'
img_url = "https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/refs/heads/main/rita_segmented.jpg"
im_array = io.imread(img_url)
imag = Image.fromarray(im_array).convert('RGB')
imag


# In[ ]:


# How the network sees the image with Imagenet transforms
plt.imshow(val_test_transforms(imag).permute(1,2,0))


# In[ ]:


# Assuming 'imag' is your PIL Image and 'val_test_transforms' is your transformation
transformed_image = val_test_transforms(imag)

# Revert the normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
reverted_image = transformed_image.permute(1, 2, 0).numpy() * std + mean
reverted_image = np.clip(reverted_image, 0, 1)  # Clip values to be in [0, 1]

# Display the reverted image
plt.imshow(reverted_image)
plt.show()


# In[ ]:


imag.save('rita.jpg')


# In[ ]:


# Assuming you have a model loaded and 'rita.jpg' is your image
prediction = predict_image(model, 'rita.jpg', val_test_transforms)
predicted_label, predicted_prob = prediction
# The model is not robust against OOD samples
oxford_pets_classes[predicted_label], predicted_prob


# ## Try a zero-shot classifier

# In[ ]:


# We can inspect the model's architecture, which is PyTorch code.
# We have downloaded both the architecture and the model's weights.
clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
clip_model


# In[ ]:


processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
processor


# In[ ]:


breed_labels.append("German Shepherd")
breed_labels.sort()


# In[ ]:


breed_labels


# In[ ]:


inputs = processor(text=breed_labels,
                   images=[imag],
                   return_tensors="pt", padding=True)
outputs = clip_model(**inputs)
logits_per_image = outputs.logits_per_image
temperature = 1.0
outputs.logits_per_image.shape


# In[ ]:


# Probabilities at different temperatures @title {run:'auto'}

slider_value = 0  # @param {type: "slider", min: 0, max: 5}
temperature = 0.7  # @param {type: "slider", min: 0, max: 2, step:0.1}

# Creating a subplot with an image in the first row and the histogram in the second row
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

axes[0].imshow(np.array(imag))
axes[0].axis('off')  # Turning off the axis for the image

probs = (logits_per_image/temperature).softmax(dim=1)

# Creating the horizontal bar plot in the second subplot
axes[1].barh(breed_labels, probs[slider_value].detach(), color='skyblue')
axes[1].set_xlim(0, 1)  # Setting the x-axis limit from 0 to 1

axes[1].set_xlabel('Probabilities')
axes[1].set_title('Probabilities of Different Labels')

plt.tight_layout()
plt.show()


# ## Create FiftyOne dataset with model predictions

# In[ ]:


# Create the 'data' folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Create the 'test' folder inside 'data' if it doesn't exist
if not os.path.exists('data/test'):
    os.makedirs('data/test')


# In[ ]:


def revert_imagenet_transforms(tensor):
  """Reverts the ImageNet transformations applied to a tensor.

  Args:
    tensor: A PyTorch float32 tensor with shape (C, H, W) that has been transformed using ImageNet normalization.

  Returns:
    A PyTorch float32 tensor with shape (C, H, W) in the range [0, 1].
  """
  # ImageNet mean and standard deviation
  imagenet_mean = np.array([0.485, 0.456, 0.406])
  imagenet_std = np.array([0.229, 0.224, 0.225])

  # Revert normalization
  reverted_tensor = tensor * torch.tensor(imagenet_std[:, None, None]).to(tensor.device) + torch.tensor(imagenet_mean[:, None, None]).to(tensor.device)

  # Clip values to [0, 1]
  reverted_tensor = torch.clamp(reverted_tensor, 0, 1)

  return reverted_tensor


# In[ ]:


def tensor_to_pil_image(tensor):
  """Converts a PyTorch float32 tensor to a PIL Image.

  Args:'
    tensor: A PyTorch float32 tensor with shape (C, H, W) in the range [0, 1].

  Returns:
    A PIL Image.
  """
  # Define the transformation to convert to PIL Image
  transform = transforms.ToPILImage()

  # Convert the tensor to a PIL Image
  image = transform(tensor)

  return image


# In[ ]:


true_labels = {}
predicted_labels = {}
predicted_probs = {}
for index, (tensor_imag, label) in tqdm(enumerate(test_dataset)):
    text_label = oxford_pets_classes[label][0]
    pil_imag = tensor_to_pil_image(revert_imagenet_transforms(tensor_imag))
    local_path = f'data/test/{text_label}_{index}.jpg'

    pil_imag.save(local_path)

    # Get predictions for the current image
    predicted_label, predicted_prob = predict_image(model, local_path, val_test_transforms)

    # Store labels and predictions in dictionaries
    true_labels[local_path] = text_label
    predicted_labels[local_path] = predicted_label
    predicted_probs[local_path] = predicted_prob



# ## Create a fiftyone `Dataset` to visualize predictions

# In[ ]:


# Delete the dataset if it exists
if fo.dataset_exists("pet_predictions"):
    fo.delete_dataset("pet_predictions")

# Create a new dataset
dataset = fo.Dataset("pet_predictions")

# Add samples to the dataset
for filepath, predicted_label in predicted_labels.items():
    sample = fo.Sample(filepath=filepath)

    # Add predicted label as a classification field
    sample["predicted_label"] = fo.Classification(
        label=oxford_pets_classes[predicted_label][0],
        confidence=predicted_probs[filepath]
    )

    # Add true label as a classification field
    sample["true_label"] = fo.Classification(
        label=true_labels[filepath],
        confidence=1.0
    )

    # Add a tag to indicate the image is part of the test set
    sample.tags.append("test sample")

    dataset.add_sample(sample)


# I want to label that the whole dataset is a test set
dataset.tags.append("test set")

# Persist the dataset to disk
dataset.persistent = True

# Print some info about the dataset
print(dataset)


# In[ ]:


session = fo.launch_app(dataset)


# ## Suggested Exercises
# 
# 1. Create a FiftyOne [`DatasetView`](https://docs.voxel51.com/user_guide/using_views.html) of all the misclassified images.
# 2. Extend the FiftyOne dataset with the images from the [Dogs vs Cats dataset on Kaggle](https://www.kaggle.com/c/dogs-vs-cats) and run your trained models on them.
# 3. Run a [zero-shot classifier](https://docs.voxel51.com/tutorials/zero_shot_classification.html) based on CLIP on the dogs vs cats dataset.
# 4. Compare our notebook with [Chapter 5 of the Deep Learning for Coders Book](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb). How does the fastai approach compare with what we did with [cyclical learning rates](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)? What would you say is missing in this notebook vs their approach?
# 5. Can you think about strategies to correlate the background of the OOD image "Rita.jpg" with the labels belong to classes that **do not** look like German Shepherds in the training set?
# 6. How would you use [Class Activation Mapping](https://voxel51.com/blog/exploring-gradcam-and-more-with-fiftyone/) to interpret the results of the model?
# 7. Look into labeling the dataset [using variants of CLIP](https://docs.voxel51.com/tutorials/zero_shot_classification.html).
# 

# In[ ]:




