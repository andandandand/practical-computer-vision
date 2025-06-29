#!/usr/bin/env python
# coding: utf-8

# # Getting Started with Image Classification and Dataset Curation using FiftyOne and PyTorch
# 
# ## Zero-shot Classification with CLIP and Supervised Learning with Convolutional Neural Networks
# 
# 
# ### Who Is this Tutorial for
# This tutorial is designed for computer vision practitioners and data scientists who want to master image classification workflows using FiftyOne. Whether you're new to computer vision or experienced with other tools, you'll learn how to leverage FiftyOne's powerful capabilities for dataset curation, model evaluation, and visual analysis.
# 
# This tutorial is appropriate for any level of computer vision knowledge. By the end of this tutorial, you'll be able to quickly identify mislabeled samples, compare classification models, create meaningful embeddings, and seamlessly move between FiftyOne and PyTorch workflows.
# 
# ### Assumed Knowledge
# We assume familiarity with basic Python programming and fundamental machine learning concepts. Knowledge of PyTorch is helpful but not required,  we'll explain the key concepts as we go. This tutorial is recommended for beginners to intermediate practitioners in computer vision.
# 
# ### Time to Complete
# 90-120 minutes
# 
# ### Required Packages
# FiftyOne, PyTorch, and several other packages are required. You can install them with:
# 
# ```bash
# pip install fiftyone==1.5.2 torch torchvision numpy
# ```
# 
# 

# 
# ## Content Overview
# 
# ### 1. MNIST Dataset Exploration with FiftyOne
# 
# Understand the MNIST dataset structure. Load the test split into FiftyOne. Compute and visualize metadata. Explore data distributions using aggregations and the FiftyOne App.
# 
# **Key concepts covered:**
# *   Loading datasets from FiftyOne Dataset Zoo
# *   Computing image metadata
# *   Using FiftyOne aggregations for data statistics
# *   Visualizing dataset distributions
# 
# ### 2. Image Embeddings with CLIP
# 
# Generate image embeddings for the test dataset using a pre-trained CLIP model. Visualize these high dimensional vectors in 2D using PCA and UMAP to understand image similarity.
# 
# **Key concepts covered:**
# *   Loading pre-trained models from FiftyOne Model Zoo
# *   Computing image embeddings with CLIP
# *   Assigning embeddings to dataset samples
# *   Dimensionality reduction: PCA and UMAP
# *   Visualizing embedding plots in FiftyOne
# 
# ### 3. Dataset Analysis using CLIP Embeddings
# 
# Continue analysis of the CLIP embeddings on the test dataset. Explore dataset clustering concepts. Compute and examine sample uniqueness and representativeness based on CLIP embeddings.
# 
# **Key concepts covered:**
# *   Introduction to clustering with embeddings
# *   Creating a similarity index
# *   Identifying outliers and representative samples
# 
# ### 4. Zero-Shot Classification with CLIP
# 
# Perform image classification on the test dataset using CLIP without task specific training. Evaluate CLIP's performance using FiftyOne's tools.
# 
# **Key concepts covered:**
# *   Zero-shot classification principles
# *   Text prompts for classification
# *   Applying a model to a FiftyOne dataset
# *   Evaluating classification results including accuracy and confusion matrix
# 
# ### 5. Supervised Classification: LeNet-5 with PyTorch
# 
# Build and train a LeNet-5 convolutional neural network from scratch using PyTorch. Prepare the MNIST training data. Implement the training loop and validation procedures.
# 
# **Key concepts covered:**
# *   LeNet-5 architecture
# *   Defining a subclass of PyTorch's `nn.Module`
# *   Splitting FiftyOne data for training and validation
# *   Creating custom PyTorch Datasets from FiftyOne views
# *   Data normalization through mean and standard deviation computation
# *   PyTorch DataLoaders
# *   Defining loss functions and optimizers
# *   Training loops and model checkpointing
# 
# ### 6. LeNet-5 Model Evaluation on Test Data
# 
# Apply the trained LeNet-5 model to the MNIST test set. Store predictions in FiftyOne. Evaluate its performance. Analyze prediction characteristics, including hardness and mistakenness.
# 
# **Key concepts covered:**
# *   Applying a PyTorch model to a FiftyOne dataset
# *   Storing predictions, confidence, and logits
# *   Evaluating classification performance
# *   Analyzing prediction confidence distributions
# *   Computing sample hardness and mistakenness
# 
# ### 7. Analysis of LeNet-5 Learned Features using Training Data
# 
# Extract embeddings from the trained LeNet-5 model using the training data. Compute and visualize these embeddings. Analyze uniqueness and representativeness based on LeNet's learned features. Identify misclassifications within the training set.
# 
# **Key concepts covered:**
# *   Extracting embeddings from intermediate PyTorch model layers
# *   Storing custom model embeddings in FiftyOne
# *   Visualizing custom embeddings with PCA and UMAP
# *   Analyzing uniqueness and representativeness of training samples
# *   Evaluating model performance on training data
# *   Identifying false positives and false negatives in training data
# 
# ### 8. Data Augmentation Concepts for MNIST
# 
# Understand effective data augmentation strategies for the MNIST dataset. Learn about geometric transformations and elastic deformations. Discuss augmentations to avoid. This section provides a conceptual discussion.
# 
# **Key concepts covered:**
# *   Rationale for data augmentation
# *   Geometric and elastic transformations suitable for MNIST
# *   Best practices and pitfalls in augmentation

# ## Installation

# In[ ]:


# Remove > /dev/null if you encounter errors during installation
get_ipython().system('pip install fiftyone==1.5.2 > /dev/null')


# ### FiftyOne Plug-ins

# We'll also install FiftyOne plugins for model evaluation and data augmentation:
# 
# 
# 
# 
# 

# In[ ]:


# Plug-in to evaluate the performance of our classification models
get_ipython().system('fiftyone plugins download      https://github.com/voxel51/fiftyone-plugins      --plugin-names @voxel51/evaluation')


# In[ ]:


# Plug-in for image augmentations
get_ipython().system('fiftyone plugins download https://github.com/jacobmarks/fiftyone-albumentations-plugin')


# ## Imports

# In[ ]:


import os

# Set environment variables for reproducibility BEFORE importing torch
os.environ['PYTHONHASHSEED'] = '51'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun
import torchvision.transforms.v2 as transforms
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from torch.utils.data import Dataset, ConcatDataset
from fiftyone import ViewField as F
import fiftyone.utils.random as four
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import albumentations as A
import cv2
import random
from typing import Optional, Dict, Tuple, Any


# ## The MNIST dataset
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/mnist_clean.png?raw=true)
# 
# The Modified National Institute of Standards and Technology (MNIST) dataset stands as one of the most influential benchmarks in computer vision and machine learning history. Created by Yann LeCun and colleagues in 1998, MNIST transformed a collection of handwritten digits from American Census Bureau employees and high school students into a standardized machine learning challenge that has shaped decades of research.
# 
# 
# 
# **Dataset Structure and Characteristics**
# 
# The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits (0-9). These are grayscale images of size 28x28 pixels, with each pixel value ranging from 0 (black) to 255 (white). The images have been size-normalized and centered, making them ideal for learning fundamental computer vision concepts without the complexity of dealing with varying scales, rotations, or backgrounds found in natural images. You can inspect the samples on the test portion of the dataset through [try.fiftyone.ai](https://try.fiftyone.ai/datasets/mnist/samples).
# 
# **Historical Significance and Impact**
# 
# MNIST earned its status as the "Hello World" of computer vision for several reasons. First, it provided the research community with a common benchmark that was computationally tractable, even early personal computers could train models on MNIST in reasonable time. Second, its simplicity allowed researchers to focus on algorithmic innovations rather than data preprocessing challenges. Landmark achievements in deep learning, from early multilayer perceptrons to convolutional architectures like LeNet-5, were first demonstrated and validated on MNIST.
# 
# The dataset served as a proving ground for fundamental concepts of modern computer vision: convolutional neural networks, regularization techniques, and optimization algorithms were all tested on these handwritten digits. Many techniques that seem obvious today, like data augmentation, dropout, and batch normalization, were first explored and validated using MNIST as a testbed.
# 
# **Why MNIST Remains Relevant**
# 
# While critics sometimes dismiss MNIST as "too easy" for modern standards, it continues to serve crucial educational and research purposes. For newcomers to computer vision, MNIST provides an ideal environment to understand core concepts without overwhelming complexity. The dataset is small enough to experiment with quickly, yet rich enough to demonstrate important phenomena like overfitting, the importance of data augmentation, and the impact of architectural choices.
# 
# Moreover, MNIST's apparent simplicity can be deceptive. Achieving state-of-the-art performance >99.7% accuracy requires sophisticated techniques and careful attention to detail, making it an interesting benchmark for testing new methodologies.
# 
# See the [discussion on Kaggle here](https://www.kaggle.com/competitions/digit-recognizer/discussion/61480) to get inspiration of how to best tackle this challenge.
# 
# 
# ## MNIST in the Modern Era
# 
# Today, MNIST serves as an excellent starting point for understanding how modern techniques like embeddings, zero-shot classification, and transfer learning work. While a model trained specifically on MNIST might achieve 99%+ accuracy, applying a general-purpose vision model like CLIP without any MNIST-specific training provides insights into how well these models generalize and what they've learned about visual patterns from their massive training datasets.
# 
# This makes MNIST perfect for comparing traditional supervised learning approaches with modern pre-trained models, helping us understand the trade-offs between task-specific optimization and general-purpose visual understanding.
# 
# 
# ## CLIP
# 
# **CLIP (Contrastive Language-Image Pre-training)** is a vision-language model developed by OpenAI that learns to understand the relationship between images and text descriptions. Traditional computer vision models are trained on fixed sets of image categories, but CLIP was trained on 400 million image-text pairs from the internet, learning to match images with their captions. This training enables CLIP to perform "zero-shot" classification: the ability to classify images into categories it has not seen during training by comparing the image representation with text descriptions of potential classes. The model works by encoding both images and text into the same high-dimensional embedding space, where similar concepts cluster together, allowing it to determine which text description best matches a given image through similarity comparison.
# 
# ![](https://github.com/andandandand/images-for-colab-notebooks/blob/main/clip%20contrastive%20pre-training.png?raw=true)
# 
# ### CLIP vs MNIST
# 
# In [OpenAI's 2021 CLIP paper](https://arxiv.org/abs/2103.00020), the comparison with MNIST revealed fascinating insights about zero-shot learning capabilities. While supervised models trained specifically on MNIST achieve near-perfect accuracy (>99.7%), OpenAI's best performing variant of CLIP (trained on about 400 million image-text pairs) achieved only 88% accuracy on these handwritten digits. This is interesting considering the model was not trained explicitly trained on any variant of this dataset. This comparison highlighted a shift in modern computer vision from specialized models that excel at narrow tasks to general models that perform reasonably well across diverse domains. The 11+ percentage point gap between supervised and zero-shot approaches on MNIST demonstrates both the power and limitations of general-purpose vision-language models, making MNIST an excellent case study for understanding the trade-offs between specialized optimization and general-purpose learning, particularly valuable for exploring modern approaches like few-shot learning, prompt engineering, and transfer learning strategies.
# 
# Note that the variant of CLIP that we use in our experiments of this notebook, `"clip-vit-base32-torch"` (a Vit-B/32 model) is *not* the top performing variant of CLIP that is showcased on the original OpenAI paper. That would be "ViT-L/14@336px", a bigger vision transformer model. The Vit-B/32 base models remain interesting and widely used due to their lower number of parameters.
# 
# 
# **Known Issues and Research Opportunities**
# 
# The original dataset contains several images where the ground truth labeling is ambiguous or questionable. The academic community has identified [several of these ambiguous samples](https://arxiv.org/abs/1912.05283). These labeling inconsistencies, while representing less than 0.1% of the dataset, provide opportunities to explore data quality assessment techniques.
# 
# Finding and analyzing these edge cases teaches valuable lessons about real-world data challenges. In production systems, you'll inevitably encounter ambiguous samples, annotation errors, and edge cases. MNIST's imperfections make it an excellent sandbox for developing robust approaches to handle these issues. Let's see if we can find them using FiftyOne's powerful analysis capabilities!

# ## Zero-shot Classification vs "Traditional" Supervised Training
# 
# **Zero-shot classification** leverages pre-trained models like CLIP that have learned rich visual representations from massive datasets, allowing them to classify images into "interpolated" categories they've not explicitly seen during training. These models understand the semantic relationship between images and text descriptions, enabling classification through natural language prompts like "a photo of the digit 3" without requiring any task-specific training data. In contrast, **traditional supervised training** requires labeled examples for each class you want to predict. You must provide the model with many of images of each digit along with their correct labels, then train the network to learn the mapping from pixel patterns to class labels through backpropagation. While supervised training often achieves higher accuracy on specific datasets and allows for domain-specific optimization, zero-shot approaches offer remarkable flexibility and can instantly work on new classification tasks without additional training time or computational resources.

# ![](https://github.com/andandandand/images-for-colab-notebooks/blob/main/clip%20zero-shot%20prediction.png?raw=true)

# ## Where CLIP Fails: Specialized Domains
# 
# **Understanding the Limitations of General-Purpose Vision Models**
# 
# While CLIP excels at understanding natural images and common objects, it struggles with specialized domains that differ from its internet-scale training data. MNIST handwritten digits represent one such domain where CLIP's performance drops compared to task-specific models.
# 
# **Domain-Specific Challenges:**
# 
# **Limited Training Exposure**: CLIP trained on web-scraped image-text pairs, which contain fewer examples of handwritten digits compared to photographs of everyday objects. The model lacks deep exposure to the subtle variations in handwriting styles.
# 
# **Scale and Context Mismatch**: CLIP expects high-resolution, colorful images with rich contextual information. MNIST's 28x28 grayscale digits provide minimal visual context that CLIP relies on for classification decisions.
# 
# **Specialized Visual Features**: Handwritten digits require recognition of fine-grained stroke patterns, curves, and connections that differ from the broader visual concepts CLIP learned from natural images. A "6" versus "9" distinction depends on subtle orientation cues.
# 
# **Text-Image Alignment**: CLIP's strength lies in matching images with descriptive captions. Simple digit classification lacks the rich semantic relationships between visual and textual information that CLIP exploits in other domains.
# 
# **Performance Gap**: This explains why CLIP achieves only ~88% accuracy on MNIST while specialized CNNs reach >99%. The 11+ percentage point gap highlights the trade-off between general-purpose capabilities and domain-specific optimization.
# 
# Other specialized domains where CLIP struggles include medical imaging, satellite imagery, microscopy, and industrial inspection: areas where domain expertise and task-specific training data prove more valuable than general visual understanding.
# 
# 
# 

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/clip_limitations.png?raw=true)

# ### Loading the MNIST Dataset from FiftyOne's Dataset Zoo
# 
# A FiftyOne dataset wraps together the annotations and image data into a unified, queryable structure that makes computer vision workflows seamless. Unlike traditional approaches where you might manage separate files for images and labels, FiftyOne treats each sample as a rich object containing the image itself, ground truth labels, metadata, and any predictions or embeddings you add later. This design enables powerful operations like filtering by class imbalance, visualizing prediction confidence, or finding samples with specific characteristics, all through a consistent API.
# 
# Loading MNIST from [FiftyOne's Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html) is straightforward:
# 

# In[ ]:


# We will load the test split from the dataset first
test_dataset = foz.load_zoo_dataset("mnist", split='test')
test_dataset


# We launch the FiftyOne app to visualize the test set.

# In[ ]:


session = fo.launch_app(test_dataset, auto=False)


# With `compute_metadata()` we add the size in bytes, the image file type, the width and height of the image, and the number of channels to our dataset.

# In[ ]:


test_dataset.compute_metadata()


# We can do [aggregations](https://docs.voxel51.com/user_guide/using_aggregations.html) on the dataset to explore the dataset.
# 
# 

# We can use the [bounds](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.bounds) aggregation to compute the [min, max] range of a numeric field of a dataset. And [mean()](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.bounds) and [std()](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.std) to compute the mean and standard deviation of it.
# 
# 

# In[ ]:


test_dataset.bounds("metadata.size_bytes")


# In[ ]:


test_dataset.mean("metadata.size_bytes"), test_dataset.std("metadata.size_bytes")


# Try filtering by label and visualizing the metadata of the MNIST images through the FiftyOne app.
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/filtering_by_label_mnist_w_metadata.png?raw=true)

# In[ ]:


session.refresh()
print(session.url)


# We can also visualize the distributions of `metadata.size_bytes` and `ground_truth.label`. Both of these are relevant when we decide the batch size on which to run our models and to evaluate the balance of classes in our dataset.
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/distribtution_size_bytes.png?raw=true)
# 
# For this try clicking on the `+` symbol next to samples, select `Histograms` and then `metadata.size_bytes` and `ground_truth.label` from the dropdown menu. The `Split horizontally button` will allow you to see the panel alongside the image data.
# 
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/ground_truh_distribution_mnist.png?raw=true)

# ## Creating Image Embeddings with CLIP
# 
# **Image embeddings** are high-dimensional vector representations that capture the semantic and aesthetic content of images in a vector format that machine learning models can understand and compare.
# 
# Think of embeddings as a way to translate visual concepts into vectors where similar images will correspond to similar embedding vectors, while visually or semantically different images will have more distant vectors in a high-dimensional space.
# 
# OpenAI's CLIP model creates particularly powerful embeddings because it was trained to understand the relationship between images and their matching text descriptions, enabling it to capture rich semantic meaning. In FiftyOne, creating embeddings with CLIP is straightforward:
# 
# 1. We obtain the model through [`foz.load_zoo_model(""clip-vit-base32-torch"")`](https://docs.voxel51.com/model_zoo/models.html#clip-vit-base32-torch) and pass it to the GPU (if we have it available).
# 2. We use the [`compute_embeddings()`](https://docs.voxel51.com/api/fiftyone.brain.internal.core.elasticsearch.html#fiftyone.brain.internal.core.elasticsearch.ElasticsearchSimilarityIndex.compute_embeddings) method:
# 
# 
# 

# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = foz.load_zoo_model("clip-vit-base32-torch",
                                device=device)
print(f"The model is loaded on {clip_model._device}")


# In[ ]:


# Calculate the total number of parameters in the model
total_params = sum(p.numel() for p in clip_model._model.parameters())

print(f"The CLIP model has {total_params:,} parameters.")


# This will take about 3 min on a Google Colab instance with GPU enabled

# In[ ]:


clip_embeddings = test_dataset.compute_embeddings(model=clip_model,
                                        batch_size=512,
                                        num_workers=2)


# The collection of embeddings is a NumPy array. Each embedding is a 512-dimensional vector. We can think about each embedding as a distinct point in a high-dimensional space.

# In[ ]:


# Check the format and shape of our embeddings vector
type(clip_embeddings), clip_embeddings.shape


# Here we attach each embedding to its corresponding image sample.

# In[ ]:


for index, sample in enumerate(test_dataset):
    sample["clip_embeddings"] = clip_embeddings[index]
    sample.save()


# We see that the first sample now has the field `embeddings` attached to it, and we can query its shape.
# 

# In[ ]:


test_dataset.first().clip_embeddings.shape


# ## Creating a Similarity Index
# 
# A **similarity index** allows for efficient searching of similar samples within a dataset based on their embeddings. Instead of comparing a query sample's embedding to every other embedding in the dataset (which can be slow for large datasets), the index organizes the embeddings in a way that allows for fast retrieval of the most similar samples. This is particularly useful for tasks like:
# 
# - Finding visually similar images.
# - Identifying near-duplicate samples.
# - Exploring clusters of similar data points.
# 
# In this case, we're creating a similarity index based on the CLIP embeddings, which capture the semantic content of the images. This index will enable us to quickly find images that are semantically similar to a given query image.
# 
# 

# In[ ]:


# Compute similarity index based on CLIP embeddings
print("Computing similarity index based on CLIP embeddings...")

similarity_index = fob.compute_similarity(
    test_dataset,
    embeddings="clip_embeddings",  # Field containing the CLIP embeddings
    brain_key="clip_cosine_similarity_index",  # Unique identifier for this similarity index
    backend="sklearn",  # Can also use "pinecone" for large datasets
    metric="cosine"  # Similarity metric (cosine is good for embeddings)
)

print(f"Similarity index computed successfully!")


# In[ ]:


# Display similarity index information
print(f"Similarity Index Details:")
print(f"- Total samples indexed: {len(test_dataset)}")
print(f"- Embeddings field: {similarity_index.config.embeddings_field}")
print(f"- Metric: {similarity_index.config.metric}")

# Example 1: Find similar images to a specific digit
query_sample = test_dataset.first()
print(f"""\nQuerying for images similar to sample: {query_sample.id}
            with label {query_sample.ground_truth.label}""")

# Use the sample ID (string) instead of the sample object
similar_view = test_dataset.sort_by_similarity(
    query_sample.id,  # Pass the sample ID, not the sample object
    brain_key="clip_cosine_similarity_index",
    k=10,  # Return top 10 most similar
    reverse=False  # Most similar first
)

print(f"Found {len(similar_view)} most similar samples")
session.view = similar_view
session.refresh()
print(session.url)


# ### Creating a 2D Projection of the Embeddings for Visualization
# 
# **Compressing High-Dimensional Representations for Visual Analysis**
# 
# Projecting high-dimensional embeddings to 2D enables visual exploration of data structure and relationships. This process represents lossy compression. The 2D visualization cannot capture all information present in the original embedding space. The projection serves as an approximation for understanding data patterns, not for precise analysis.
# 
# **Dimensionality Reduction Methods**: PCA and UMAP compress embeddings through different approaches. PCA finds linear combinations that preserve maximum variance, making it faster and deterministic, the same data produces identical results. UMAP uses sophisticated manifold learning to preserve local neighborhoods and reveal cluster structure, but operates through stochastic processes that can produce different results across runs.
# 
# **Visualization Purpose**: These projections help identify clusters, outliers, and data distribution patterns. Points close together in the visualization often share similar characteristics, while distant points represent different concepts. The projection quality depends on how well the chosen method preserves the relationships present in the original high-dimensional space.
# 
# **Interpretation Limits**: Remember that 2D projections cannot represent all relationships from the original embeddings. Some distances may appear closer or farther than they actually are in the full dimensional space.
# 
# 

# In[ ]:


pca_visualization = fob.compute_visualization(test_dataset,
                                              method="pca",
                                              embeddings="clip_embeddings",
                                              num_dims=2,
                                              brain_key="pca_visualization_clip_embeds")


# In[ ]:


umap_visualization = fob.compute_visualization(test_dataset,
                                              method="umap",
                                              embeddings="clip_embeddings",
                                              num_dims=2,
                                              brain_key="umap_visualization_clip_embeds")


# ## Visualizing Embeddings in the FiftyOne App
# 
# To view the embedding visualizations (PCA and UMAP) in the FiftyOne App:
# 
# 1. **Open the Panel**: Click on the "**+**" icon next to "Samples" in the FiftyOne App's header.
# 2. **Select Visualizations**: From the dropdown menu, select "Embeddings" under the "Visualizations" section.
# 3. **Choose Embeddings**: In the Embeddings panel that appears, select the embedding field you want to visualize from the dropdown (e.g., `pca_visualization_clip_embeds` or `umap_visualization_clip_embeds`).
# 
# You can then interact with the 2D plot, hovering over points to see the corresponding images and selecting regions to filter the samples in the grid.

# In[ ]:


session.refresh()
print(session.url)


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/image_embeddings_zero_cluster.gif?raw=true)

# Let's use the FiftyOne app to explore the  samples in the dataset.

# In[ ]:


session.refresh()
print(session.url)


# ## Clustering through Embeddings

# 
# **Clustering** transforms the high-dimensional embedding space into groups of similar images, helping us discover hidden patterns and structure in our dataset. Rather than manually browsing through thousands of images, clustering algorithms like K-means, HDBSCAN, or Gaussian Mixture Models automatically identify samples that share visual or semantic characteristics. This is particularly powerful when combined with CLIP embeddings, as the clusters often correspond to semantic concepts like "outdoor scenes," "close-up portraits," or "nighttime images." FiftyOne provides a dedicated clustering plugin that makes this analysis seamless. Install it with:
# 
# 
# 

# In[ ]:


get_ipython().system('fiftyone plugins download https://github.com/jacobmarks/clustering-plugin')


# Once installed, you can access clustering functionality directly from the FiftyOne App by pressing the backtick key (`) and typing `compute_clusters`. The plugin offers multiple clustering algorithms and customizable parameters. This workflow is valuable for understanding the groupings within your data produced by the embedding model. Learn more about clustering workflows and advanced techniques in the [FiftyOne clustering tutorial](https://docs.voxel51.com/tutorials/clustering.html).
# 

# In[ ]:


session.refresh()
print(session.url)


# ## Zero-shot Classification

# Now that we understand how CLIP creates meaningful embeddings, we can leverage these representations for **zero-shot classification** - the ability to classify images into categories without any task-specific training. CLIP's vision-language training enables it to understand the semantic relationship between images and text descriptions, allowing us to classify MNIST digits simply by providing natural language descriptions of each class.
# 
# The key insight is that CLIP learns to map both images and text into the same embedding space, where semantically similar concepts cluster together. By computing the similarity between an image's embedding and text embeddings for each class description, we can determine which category best matches the input image. This approach is remarkably flexible, we can change our classification categories simply by modifying the text prompts, without retraining the models.
# 
# For MNIST digit classification, we'll use descriptive text prompts combined with the written names of each digit.
# 

# In[ ]:


# We obtain the distinct labels of the dataset
dataset_classes = sorted(test_dataset.distinct("ground_truth.label"))
dataset_classes


# ## The Effect of the Text Prompt

# With CLIP, the `text_prompt` parameter can have a significant effect on the accuracy of the model, as it affects the text embedding for the label.
# 
# ### Experiment with the Prompt
# 
# Try modifying the `text_prompt` field with:
# 
# * "The handwritten digit "
# * "A grayscale image of the number "
# * "A pixel illustration of the digit "
# * "A low resolution image of the number"
# * "An MNIST digit "
# 
# And observe the effect on the accuracy, precision and recall on our classification report.

# In[ ]:


clip_model = foz.load_zoo_model(
    "clip-vit-base32-torch",
    text_prompt="A photo of ",
    classes=dataset_classes,
    device=device
)


# In[ ]:


# The CLIP model preprocesses the input data
clip_model.preprocess


# Notice that the CLIP model was originally trained on much larger images and that
# these where RGB. In order to make the MNIST grayscale image data work with CLIP, we transform it scale it using the mean and standard deviations of the original training data.

# In[ ]:


clip_model._transforms


# The `store_logits=True` parameter is important as it preserves the model's confidence scores for each prediction, enabling us to analyze prediction uncertainty and identify samples where the model was unsure. After applying the model, we can compute comprehensive evaluation metrics and visualize the results using FiftyOne's evaluation framework:

# In[ ]:


test_dataset.apply_model(
    model=clip_model,
    label_field="clip_zero_shot_classification",
    # We need to store the logits to compute the "mistakenness" value
    store_logits=True,
    # This is how many samples we will show to the model at once
    batch_size=256,
    progress_bar=True,
     # Use more running threads if you have multiple cpus
    num_workers=os.cpu_count()
)


# In[ ]:


session.refresh()
print(session.url)


# ## Evaluating CLIP's Classification Performance
# 
# With our zero-shot CLIP predictions generated, we need to evaluate how well the model performs compared to the ground truth labels. FiftyOne provides powerful evaluation capabilities that go beyond simple accuracy metrics, allowing us to understand where and why our model succeeds or fails.
# 
# [FiftyOne's Model Evaluation Panel](https://docs.voxel51.com/user_guide/app.html#model-evaluation-panel-sub-new ) provides interactive confusion matrices, per-class metrics, and the ability to drill down into specific failure modes. This visual analysis helps identify systematic errors, class imbalances, and edge cases that pure numerical metrics might miss, enabling targeted improvements to your classification pipeline.
# 
# 
# 

# In[ ]:


clip_evaluation_results = test_dataset.evaluate_classifications(
    "clip_zero_shot_classification",
    gt_field="ground_truth",
    eval_key="clip_zero_shot_eval")

session.refresh()
print(session.url)


# In[ ]:


clip_evaluation_results.print_report(classes=dataset_classes, digits=3)


# In[ ]:


# The confusion matrix has a lot activity outside of the main diagonal
clip_evaluation_results.plot_confusion_matrix()


# ## Inspect the Images where the Model is Performing Worst
# 
# We can create a view of the images labeled as nines on the dataset, to get a closer look at what CLIP is misclassifying.  

# In[ ]:


nines_view = test_dataset.filter_labels("ground_truth",
                                        F("$ground_truth.label") == "9 - nine"
                                       )


# #### The Purpose of `session.refresh()` in FiftyOne
# 
# The `session.refresh()` command in FiftyOne syncs the state of the FiftyOne App interface with the state of our FiftyOne dataset object in the Python kernel.
# 
# When we modify a dataset or view in your Python code by adding new fields, computing embeddings, applying models, filtering samples, or adding tags, these changes are made to the dataset object in your Python session. However, the FiftyOne App, which runs in a separate process, doesn't automatically know about these updates.
# 
# Calling `session.refresh()` explicitly tells the FiftyOne App to reload the current view of the dataset from the backend. This ensures that the App's visualization and sample grid reflect the latest changes you've made in your notebook, allowing you to see and interact with the results of your code in the App's UI. Without `session.refresh()`, the App might show an outdated version of your dataset.

# In[ ]:


session.view = nines_view
session.refresh()
print(session.url)


# ## Clearing the CLIP Model from GPU memory
# 
# After using the CLIP model, it's good practice to free up the GPU memory it occupied. This is done by:
# 
# 1. Deleting the model variable (`del clip_model`).
# 2. Running Python's garbage collector (`gc.collect()`) to clean up references.
# 3. Emptying the CUDA cache (`torch.cuda.empty_cache()`) to release cached memory on the GPU.
# 
# This ensures that the GPU memory (aka VRAM) is available for the next batches of images and models.
# 

# In[ ]:


# Delete the model variable
del clip_model

# Run Python's garbage collector
gc.collect()

# Empty the CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("CUDA device memory from clip_model should be cleared.")


# ## Reproducibility for Training Experiments
# 
# To ensure full reproducibility of your training experiments, you need to set random seeds for all libraries and operations that involve randomness. This ensures that initial model weights, data shuffling, and any other random operations are the same across different runs.
# 
# 

# In[ ]:


def set_seeds(seed=51):
    """
    Set seeds for complete reproducibility across all libraries and operations.

    Args:
        seed (int): Random seed value
    """
    # Set environment variables before other imports
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # CUDA deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # OpenCV
    cv2.setRNGSeed(seed)

    # Albumentations (for data augmentation)
    try:
        A.seed_everything(seed)
    except AttributeError:
        # Older versions of albumentations
        pass

    # PyTorch deterministic algorithms (may impact performance)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Some operations don't have deterministic implementations
        print("Warning: Some operations may not be deterministic")

    print(f"All random seeds set to {seed} for reproducibility")



# Usage: Call this function at the beginning and before each training phase
set_seeds(51)

# Additional reproducibility considerations:

def create_deterministic_training_dataloader(dataset, batch_size, shuffle=True, **kwargs):
    """
    Create a DataLoader with deterministic behavior.

    Args:
        dataset: PyTorch Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        **kwargs: Additional DataLoader arguments

    Returns:
        Training DataLoader with reproducible behavior
    """
    # Use a generator with fixed seed for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(51)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        **kwargs
    )

# Example usage:
# train_loader = create_deterministic_dataloader(
#     torch_train_set,
#     batch_size=64,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True
# )



# ## Creating a Custom Convolutional Neural Networks in PyTorch (Two Versions of LeNet-5)
# 
# ![](https://raw.githubusercontent.com/andandandand/practical-computer-vision/refs/heads/main/images/lenet5-architecture.png)
# 
# While zero-shot classification with CLIP demonstrates the power of modern pre-trained models, understanding how to build and train convolutional neural networks from scratch remains fundamental to computer vision. **LeNet-5**, proposed by Yann LeCun in 1998, represents one of the earliest and most influential CNN architectures. Despite its age, LeNet-5 perfectly illustrates core CNN concepts including convolutional layers, pooling layers, and the transition from feature extraction to classification.
# 
# LeNet-5's architecture is elegantly simple yet effective: it uses alternating convolutional and pooling layers to extract hierarchical features, followed by fully connected layers for classification. The network learns low-level features like edges and curves in early layers, then combines these into higher-level digit patterns in deeper layers. This hierarchical feature learning principle underlies virtually all modern CNN architectures.
# 
# For MNIST digit classification, LeNet-5 provides an excellent baseline to compare against CLIP's zero-shot performance. While CLIP leverages massive-scale pre-training and vision-language understanding, LeNet-5 demonstrates what's possible with task-specific supervised learning on a much smaller scale. Building this model from scratch in PyTorch teaches essential concepts about gradient-based optimization, backpropagation, and the relationship between network architecture and learning capacity.
# 
# 

# Here is [a great animation](https://youtu.be/UxIS_PoVoz8?si=3ibZms7Hk1oSj55k) showcasing the architecture.
# 
# 

# In[ ]:


class ClassicLeNet5(nn.Module):
    """
    LeNet-5 CNN architecture for MNIST digit classification.

    Original paper: "Gradient-Based Learning Applied to Document Recognition"
    by LeCun et al. (1998)

    Architecture (maintains original design with padding):
    Input (28x28) -> Pad to (32x32) -> C1 (6@28x28) -> S2 (6@14x14) ->
    C3 (16@10x10) -> S4 (16@5x5) -> C5 (120@1x1) -> F6 (84) -> Output (10)
    """

    def __init__(self, num_classes=10):
        super(ClassicLeNet5, self).__init__()

        # Feature extraction layers
        # C1: Convolutional layer - 6 feature maps, 5x5 kernels
        # Add padding=2 to convert 28x28 input to 32x32, maintaining original design
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)

        # S2: Subsampling layer (average pooling) - 2x2 with stride 2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: Convolutional layer - 16 feature maps, 5x5 kernels
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # S4: Subsampling layer (average pooling) - 2x2 with stride 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: Convolutional layer - 120 feature maps, 5x5 kernels (original design)
        # This reduces the 5x5 feature maps to 1x1
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)

        # Classification layers
        # F6: Fully connected layer with 84 units
        self.fc1 = nn.Linear(120, 84)

        # Output layer: 10 classes for digits 0-9
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # C1: Convolution + activation (padding converts 28x28 to 32x32, then conv to 28x28)
        # Input: (batch, 1, 28, 28) -> Pad to (32, 32) -> Conv to (batch, 6, 28, 28)
        x = torch.tanh(self.conv1(x))

        # S2: Average pooling
        # Input: (batch, 6, 28, 28) -> Output: (batch, 6, 14, 14)
        x = self.pool1(x)

        # C3: Convolution + activation
        # Input: (batch, 6, 14, 14) -> Output: (batch, 16, 10, 10)
        x = torch.tanh(self.conv2(x))

        # S4: Average pooling
        # Input: (batch, 16, 10, 10) -> Output: (batch, 16, 5, 5)
        x = self.pool2(x)

        # C5: Convolution + activation (original 5x5 kernel design)
        # Input: (batch, 16, 5, 5) -> Output: (batch, 120, 1, 1)
        x = torch.tanh(self.conv3(x))

        # Flatten for fully connected layers
        # Input: (batch, 120, 1, 1) -> Output: (batch, 120)
        x = x.view(x.size(0), -1)

        # F6: Fully connected + activation
        # Input: (batch, 120) -> Output: (batch, 84)
        x = torch.tanh(self.fc1(x))

        # Output layer (no activation - raw logits)
        # Input: (batch, 84) -> Output: (batch, 10)
        x = self.fc2(x)

        return x


# Below is an alternative and more modern implementation. Here the activation functions have been switched from tanh to ReLU and Max Pooling is used instead of Average Pooling. Feel free to choose either!
# 
# Unlike the CLIP model, that has been pretrained, these networks are trained from scratch. We will use train portion of the MNIST dataset to do this.

# In[ ]:


# Alternative modern version with ReLU and MaxPooling
class ModernLeNet5(nn.Module):
    """
    Modernized version of LeNet-5 with ReLU activations and max pooling.
    Often performs better on MNIST than the original version.
    """

    def __init__(self, num_classes=10):
        super(ModernLeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(Fun.relu(self.conv1(x)))
        x = self.pool(Fun.relu(self.conv2(x)))
        x = Fun.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = Fun.relu(self.fc1(x))
        x = self.dropout(x)  # Add dropout for regularization
        x = self.fc2(x)

        return x


# ## Obtain the Training Dataset from FiftyOne's Zoo

# In[ ]:


# We use the training split to train our LeNet model
# We make this dataset persistent as we want to save our changes for multiple sessions
train_val_dataset = foz.load_zoo_dataset("mnist",
                                         split='train',
                                         dataset_name="mnist-train-val",
                                         persistent=True)

train_val_dataset.compute_metadata()


# ## Splitting into Train and Validation
# 
# The **validation set** serves as an unbiased evaluation mechanism during model development, acting as a proxy for real-world performance before touching the final test set. While the training set teaches the model to recognize patterns in handwritten digits, the validation set reveals whether the model has truly learned generalizable features or simply memorized the training data, a phenomenon known as overfitting.
# 
# During training, we monitor both training and validation **loss** simultaneously. **Loss** is a numerical measure of how far off the model's predictions are from the correct answers - lower loss means better performance. We use categorical cross-entropy loss, which penalizes confident wrong predictions more heavily than uncertain ones. **Training loss** typically decreases steadily as the model learns, but **validation loss** tells the real story. If validation loss plateaus or begins increasing while training loss continues decreasing, the model is overfitting and memorizing training-specific details rather than learning robust digit recognition patterns. This signals when to stop training, adjust hyperparameters, or modify the architecture.
# 
# The validation set also enables **hyperparameter tuning** without contaminating our final evaluation. We can experiment with different learning rates, batch sizes, regularization techniques, or architectural modifications, using validation loss to guide these decisions. Each configuration gets evaluated on the same held-out validation data, ensuring fair comparisons.
# 
# **Important: the test set remains completely untouched** throughout the development process. Only after we've selected our final model configuration based on validation performance do we evaluate on the test set once, giving us an honest estimate of how the model will perform on truly unseen data. This three-way split (train/validation/test) is fundamental to responsible machine learning development and prevents the subtle "data leakage" that can make models appear better than they actually are.

# In[ ]:


# The images come with the 'train' tag and this must be deleted
# at the sample level.
train_val_dataset.untag_samples(["train", "validation"])


# In[ ]:


set_seeds(51)
# Create random 85%/15% split using tags
four.random_split(train_val_dataset,
                  {"train": 0.85, "validation": 0.15},
                  # The seed makes the split reproducible
                  seed=51)

# Verify the split by counting tags
tag_counts = train_val_dataset.count_sample_tags()
print("Tag counts after split:")
print(tag_counts)

# Separate validation and train FO datasets
train_dataset = train_val_dataset.match_tags("train").clone()
val_dataset = train_val_dataset.match_tags("validation").clone()

# Set names for FO datasets using the 'name' property
train_dataset.name = "mnist-training-set"
val_dataset.name = "mnist-validation-set"

# Define persistency
train_dataset.persistent = True
val_dataset.persistent = True

# Verify no overlap between train and validation
train_ids = set(train_dataset.values("id"))
val_ids = set(val_dataset.values("id"))
overlap = train_ids.intersection(val_ids)


# ## Moving the FiftyOne Data Splits to torch Datasets
# 
# To train our PyTorch model, we need to convert our FiftyOne dataset views into PyTorch `Dataset` objects that can load and preprocess images during training. This bridge between FiftyOne's dataset management and PyTorch's training pipeline is important for maintaining both the metadata and annotations while enabling batch processing. PyTorch's `Dataset` class provides a standardized interface for data loading, handling tasks like image loading, preprocessing transforms, and label conversion. By creating a custom dataset class that works with FiftyOne's file paths and labels, we can leverage PyTorch's `DataLoader` for batching, shuffling, and parallel data loading while preserving all the dataset analysis capabilities that FiftyOne provides. This approach allows us to move between FiftyOne's visual analysis and PyTorch's training workflows without duplicating data or losing the rich metadata we've computed.

# In[ ]:


# Custom PyTorch Dataset class for MNIST training data
class CustomTorchImageDataset(torch.utils.data.Dataset):
    def __init__(self, fiftyone_dataset,
                 image_transforms=None,
                 label_map=None,
                 gt_field="ground_truth"):
        self.fiftyone_dataset = fiftyone_dataset
        self.image_paths = self.fiftyone_dataset.values("filepath")
        self.str_labels = self.fiftyone_dataset.values(f"{gt_field}.label")
        self.image_transforms = image_transforms

        if label_map is None:
            self.label_map = {str(i): i for i in range(10)}  # "0"->0, "1"->1, etc.
        else:
            self.label_map = label_map

        print(f"CustomTorchImageDataset initialized with {len(self.image_paths)} samples.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('L')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.randn(1, 28, 28), torch.tensor(-1, dtype=torch.long)

        if self.image_transforms:
            image = self.image_transforms(image)

        label_str = self.str_labels[idx]
        label_idx = self.label_map.get(label_str, -1)
        if label_idx == -1:
            print(f"Warning: Label '{label_str}' not in label_map for image {image_path}")

        return image, torch.tensor(label_idx, dtype=torch.long)


# ## Computing the Mean and Standard Deviation
# 
# Before training neural networks, we compute the **mean and standard deviation** of our input data to apply **standard scaling** (also called normalization or standardization).
# 
# 
# **Standard Scaling Formula:**
# 
# $$z = \frac{x - \mu}{\sigma}$$
# 
# Where:
# - $z$ = standardized value
# - $x$ = original pixel value
# - $\mu$ = mean of all pixel values in the dataset
# - $\sigma$ = standard deviation of all pixel values in the dataset
# 
# In PyTorch, this is implemented with torch.`transforms.Normalize((mean_intensity), (std_intensity))`
# 
# This preprocessing step transforms our pixel values to have zero mean and unit variance, which provides several critical benefits:
# 
# **Why Standard Scaling Matters:**
# 
# **Gradient Optimization**: Neural networks learn through gradient descent, which works best when input features are on similar scales. Without scaling, features with larger magnitudes (like raw pixel values 0-255) can dominate the gradient updates, leading to slower convergence and unstable training.
# 
# **Weight Initialization Compatibility**: Modern weight initialization schemes (Xavier, He initialization) assume inputs are roughly centered around zero with unit variance. Standard scaling ensures our data matches these assumptions, preventing vanishing or exploding gradients during early training.
# 
# **Activation Function Efficiency**: Many activation functions (tanh, sigmoid) work optimally when inputs are centered around zero. Scaled inputs help neurons operate in the most responsive regions of these functions rather than saturating in flat regions.
# 
# **Learning Rate Stability**: With standardized inputs, we can use higher learning rates without instability, as the optimization landscape becomes more uniform across different dimensions.
# 
# For MNIST images, we transform raw pixel values from the range [0, 255] to approximately [-1, 1] with mean ≈ 0, creating a more favorable training environment that typically results in faster convergence and better final performance.
# 

# In[ ]:


def compute_stats_fiftyone(fiftyone_view):
    """
    Compute stats directly from FiftyOne using aggregations.
    Requires images to be loaded as arrays.
    """

    print("Computing image intensity statistics from FiftyOne view...")

    # Get all image filepaths
    filepaths = fiftyone_view.values("filepath")

    # Load all pixel values
    all_pixels = []

    for filepath in tqdm(filepaths):

        try:
            # Load image as grayscale array
            image = Image.open(filepath).convert('L')
            # Scale values to the range [0, 1]
            pixels = np.array(image, dtype=np.float32) / 255.0
            all_pixels.append(pixels.flatten())
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    # Concatenate all pixel values
    all_pixels = np.concatenate(all_pixels)

    # Compute statistics
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)

    print(f"Computed from {len(filepaths)} images")
    print(f"Total pixels: {len(all_pixels):,}")

    return mean, std

mean_intensity, std_intensity = compute_stats_fiftyone(train_dataset)
f"Mean: {mean_intensity:.4f}, Std: {std_intensity:.4f}"


# In[ ]:


# Map the string labels to numerical values (we need this for the PyTorch dataset)
label_map = {string_label: index for index, string_label in enumerate(dataset_classes)}
label_map


# Transform the PIL images into PyTorch tensors with scaling based on stats from the training set.

# In[ ]:


image_transforms = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((mean_intensity,), (std_intensity,))
])


# In[ ]:


torch_train_set = CustomTorchImageDataset(train_dataset,
                                          label_map=label_map,
                                          image_transforms=image_transforms)


# In[ ]:


torch_val_set = CustomTorchImageDataset(val_dataset,
                                     label_map=label_map,
                                     image_transforms=image_transforms)


# ### Create PyTorch DataLoaders
# 
# DataLoaders wrap our custom datasets and handle the mechanics of training: batching samples together, shuffling data between epochs, and loading images in parallel using multiple CPU cores. The batch size determines how many images the model processes at once, affecting both memory usage and training dynamics. We shuffle the training data to prevent the model from learning spurious patterns based on sample order, but keep validation data unshuffled since evaluation order doesn't matter. Parallel loading with multiple workers speeds up training by preparing the next batch while the GPU processes the current one.
# 
# 

# In[ ]:


# Define batch size (you can adjust this based on your GPU memory)
batch_size = 64
num_workers = os.cpu_count()  # Number of CPU cores



# In[ ]:


train_loader = create_deterministic_training_dataloader(
    torch_train_set,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)


# In[ ]:


val_loader = torch.utils.data.DataLoader(
    torch_val_set,
    batch_size=batch_size,
    shuffle=False, # No need to shuffle validation data
    num_workers=num_workers,
    pin_memory=True
)

print("Train and validation DataLoaders created successfully.")
print(f"Train DataLoader has {len(train_loader)} batches.")
print(f"Validation DataLoader has {len(val_loader)} batches.")


# ## Instantiate the Loss Function (Categorical Cross Entropy)
# 
# **Categorical Cross Entropy** is the standard loss function for multi-class classification problems like MNIST digit recognition. It measures how far our model's predicted probability distribution is from the true distribution (one-hot encoded labels).
# 
# **Mathematical Formula:**
# $$\text{CCE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
# 
# Where:
# - $C$ = number of classes (10 for MNIST digits 0-9)
# - $y_i$ = true label (1 for correct class, 0 for others)
# - $\hat{y}_i$ = predicted probability for class $i$
# 
# **Intuitions about Cross Entropy Loss**: The loss encourages the model to output high confidence (probability close to 1.0) for the correct class and low confidence for incorrect classes.
# 
# For a perfectly correct prediction (probability = 1.0 for true class), the loss approaches 0 and gradients are small (meaning that there is little change on weights). For very wrong predictions (probability = 0.001 for the ground truth class), the loss approaches infinity, strongly penalizing confident mistakes and forcing the neural network to update its weights.
# 
# PyTorch implements a numerically stable variant of CCE based on the LogSoftMax function. You can read more about it [here](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

# In[ ]:


ce_loss = nn.CrossEntropyLoss()


# ## Prepare Training and Validation for our Custom Model
# 
# We define two functions to handle the training and validation phases of each epoch. The `train_epoch()` function puts the model in training mode, processes batches through forward passes, computes loss, and updates weights via backpropagation.

# In[ ]:


def train_epoch(model, train_loader):
  batch_losses = []
  model.train()
  for images, labels in tqdm(train_loader, desc="Training: "):
      #import pdb; pdb.set_trace()

      images = images.to(device)
      labels = labels.to(device)
      # Forward pass
      logits = model(images)
      loss_value = ce_loss(logits, labels)
      # Clear gradients from previous iteration (PyTorch accumulates by default)
      optimizer.zero_grad()
      # Computes the gradients with backpropagation
      loss_value.backward()
      # Updates the weights
      optimizer.step()

      batch_losses.append(loss_value.item())

  train_loss = np.mean(batch_losses)
  return train_loss


# The `val_epoch()` function switches the model to evaluation mode and computes validation loss without updating weights, giving us an unbiased measure of performance on held-out data. These functions return the average loss across all batches, which we'll track to monitor training progress and detect overfitting.

# In[ ]:


def val_epoch(model, val_loader):
  batch_losses = []
  model.eval()
  with torch.inference_mode():
    for images, labels in tqdm(val_loader, desc="Validation: "):
      images = images.to(device)
      labels = labels.to(device)
      # Forward pass
      logits = model(images)
      loss_value = ce_loss(logits, labels)
      batch_losses.append(loss_value.item())
  val_loss = np.mean(batch_losses)
  return val_loss


# ## Defining the Optimizer
# 
# **Configuring the Learning Algorithm**
# 
# The optimizer determines how the neural network updates its weights based on computed gradients during training. This choice affects training speed, stability, and final model performance.
# 
# **Adam (Adaptive Moment Estimation)** is a variant of gradient descent that maintains running averages of both gradients and their squared values, allowing us to adapt the learning rate based on the historical behavior of each weight.
# 
# ```python
# optimizer = Adam(model.parameters(),
#                  lr=0.003, betas=(0.9, 0.999),
#                  eps=1e-08, weight_decay=0)
# ```
# 
# The learning rate (lr=0.003) controls the step size for weight updates, while the beta settings control how much history to consider when computing the adaptive rates. This configuration provides stable training for most computer vision tasks.
# 

# In[ ]:


model = ModernLeNet5().to(device)

# Define the optimizer (variant of stochastic gradient descent)
optimizer = Adam(model.parameters(),
                 lr=0.003, betas=(0.9, 0.999),
                 eps=1e-08, weight_decay=0)


# ## Training and Checkpointing the Model
# 
# **Iterative Learning with Performance Monitoring**
# 
# Training a neural network involves showing the model batches of data, computing prediction errors, and updating weights to minimize those errors. This process continues for multiple epochs, where each epoch represents one complete pass through the entire training dataset.
# 
# **The Role of Validation-Based Checkpointing**
# 
# During training, we monitor performance on both training and validation sets. Training loss decreases as the model learns, but validation loss reveals the true generalization capability. The validation set acts as a proxy for real-world performance since the model never sees these samples during weight updates.
# 
# We save model checkpoints based on validation performance rather than training performance to prevent overfitting. A model might achieve low training loss by memorizing training examples, but this doesn't guarantee good performance on new data. By saving the model weights that achieve the best validation loss, we capture the point where the model has learned generalizable patterns without overfitting to training-specific details.
# 
# ```python
# if val_loss < best_val_loss:
#     best_val_loss = val_loss
#     best_model = model
#     torch.save(best_model.state_dict(), model_save_path)
#     print('Found and saved better weights for the model')
# ```
# 
# This checkpointing strategy ensures we retain the model configuration that will perform best on unseen test data, even if training continues and validation performance later degrades due to overfitting.

# In[ ]:


# Ensure reproducibility for the training process
set_seeds(51) # You can change this number to get different results

num_epochs = 10
train_losses = []
val_losses = []

best_val_loss = float('inf')
best_model = None

# Define the path to save the model within your hard-drive
path = Path(os.getcwd()) # Feel free to change the path

model_save_path = path / 'best_lenet.pth'

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = val_epoch(model, val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)


    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        # Save the best model
        torch.save(best_model.state_dict(), model_save_path)
        print('Found and saved better weights for the model')


# ## Visualizing Training vs Validation Losses
# 
# Plotting training and validation loss over epochs provides insights into model learning dynamics. These curves reveal whether the model is learning, overfitting, or underfitting the data.
# In healthy training, both curves decrease together, with training loss lower than validation loss. When training loss continues decreasing while validation loss plateaus or increases, this indicates overfitting where the model memorizes training data rather than learning generalizable patterns. If both curves plateau at high values, the model may be underfitting and require more capacity (e.g.more layers or more weights) or more training epochs.

# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Find the epoch with the best validation loss
best_epoch = np.argmin(val_losses) + 1 # Add 1 because epochs are 1-indexed

# Add a vertical red line at the epoch with the best validation loss
plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Val Loss Epoch {best_epoch}')
plt.legend()

plt.show()


# ## Reload the Best LeNet Model Weights
# 
# After training completes, we need to load the saved model weights that achieved the best validation performance. This ensures we use the model configuration that generalizes best to unseen data, rather than the final training state which may have overfitted.
# The reloading process involves creating a new model instance with the same architecture, then loading the saved state dictionary containing the optimal weights. We must ensure the model is moved to the correct device (CPU or GPU) and set to evaluation mode for inference.

# In[ ]:


# Define the path where the best model was saved
path = Path(os.getcwd())
model_save_path = path / 'best_lenet.pth'

# Instantiate a new model with the same architecture
# Make sure you use the same model class that was trained
loaded_model = ModernLeNet5()

# Load the saved state dictionary into the new model instance
# Make sure the model is on the correct device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_model.load_state_dict(torch.load(model_save_path, map_location=device))

# Move the model to the device
loaded_model = loaded_model.to(device)

# Set the model to evaluation mode
loaded_model.eval()

print(f"Model loaded successfully from {model_save_path}")
print(f"Model is on device: {next(loaded_model.parameters()).device}")


# ## Apply our Custom Model to the Test Set
# 
# We have trained our LeNet-5 model and need to apply it to the test set to generate predictions and evaluate its performance. This step bridges the gap between PyTorch training and FiftyOne's analysis capabilities.
# 
# **Why Store Predictions as FiftyOne Classifications?**
# 
# Instead of storing raw predictions as strings or numbers, we use FiftyOne's [`Classification`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classification) objects, which provide several key advantages:
# 
# **Structured Data Storage**: Classification objects encapsulate the predicted label, confidence score, and raw logits in a standardized format that FiftyOne understands.
# 
# **Evaluation Integration**: FiftyOne's evaluation framework (`evaluate_classifications()`) can compare Classification objects against ground truth labels, generating metrics like confusion matrices, per-class accuracy, and performance reports.
# 
# **Querying and Filtering**: With Classification objects, we can filter samples by confidence thresholds, find misclassifications, or identify uncertain predictions using FiftyOne's query language:
# 
# ```python
# # Find high-confidence predictions
# high_conf = dataset.match(F("predictions.confidence") > 0.95)
# 
# # Find misclassifications  
# errors = dataset.match(F("predictions.label") != F("ground_truth.label"))
# ```
# 
# **Visual Analysis**: The FiftyOne App can visualize Classification objects with confidence scores, making it simple to spot patterns in model behavior and identify errors.
# 
# **Model Comparison**: Storing predictions in this standardized format enables comparison between different models (like our LeNet vs. CLIP's zero-shot classification) using the same evaluation framework.
# 
# **Confidence-Based Analysis**: The embedded confidence scores allow for analysis like identifying samples where the model is uncertain, which correspond to edge cases or potential labeling errors in the dataset.
# 
# This approach transforms raw model outputs into queryable metadata that integrates with FiftyOne's computer vision workflow, enabling insights into model performance and behavior.

# In[ ]:


## Apply best_model to the test set, store logits and confidence

# Create a PyTorch Dataset for the test set
torch_test_set = CustomTorchImageDataset(test_dataset,
                                      label_map=label_map, # Use the same label map as training
                                      image_transforms=image_transforms) # Use the same transforms

# Create a PyTorch DataLoader for the test set
test_loader = torch.utils.data.DataLoader(
    torch_test_set,
    batch_size=batch_size,
    shuffle=False, # No need to shuffle test data
    num_workers=num_workers,
    pin_memory=True
)

print("Test DataLoader created successfully.")
print(f"Test DataLoader has {len(test_loader)} batches.")

# Set the loaded model to evaluation mode
loaded_model.eval()

# Lists to store predictions and logits
predictions = []
all_logits = []

# Run inference on the test set
print("Applying best LeNet model to the test set...")
with torch.inference_mode(): # Disable gradient calculation
    for images, _ in tqdm(test_loader):
        images = images.to(device)

        # Forward pass to get logits
        logits = loaded_model(images)
        all_logits.append(logits.cpu().numpy()) # Store logits

        # Get predicted class indices
        _, predicted = torch.max(logits.data, 1)
        predictions.extend(predicted.cpu().numpy()) # Store predictions

# Concatenate logits from all batches
all_logits = np.concatenate(all_logits, axis=0)

print("Inference on test set complete.")
print(f"Shape of collected logits: {all_logits.shape}")
print(f"Number of collected predictions: {len(predictions)}")

# Store the predictions and logits back into the FiftyOne dataset as Classification objects
print("Storing predictions and logits as FiftyOne Classifications...")

for i, sample in enumerate(test_dataset):
    # Get the predicted class index and corresponding class name
    predicted_idx = predictions[i]
    predicted_label = dataset_classes[predicted_idx]

    # Get logits for this sample
    sample_logits = all_logits[i]

    # Calculate confidence scores (softmax applied to logits)
    confidences = Fun.softmax(torch.tensor(sample_logits), dim=0).numpy()
    predicted_confidence = float(confidences[predicted_idx])

    # Create FiftyOne Classification object with prediction
    classification = fo.Classification(
        label=predicted_label,
        confidence=predicted_confidence,
        logits=sample_logits.tolist()  # Store raw logits
    )

    # Store the Classification object in the sample
    sample["lenet_classification"] = classification

    sample.save()




# ### Verify the Stored Data Structure for Predictions
# We should see `Classification` objects with label, confidence, and logits fields. We need these to perform analytics on our FiftyOne dataset.
# 

# In[ ]:


print("\n=== Verification ===")
sample = test_dataset.first()
print(f"Sample prediction type: {type(sample.lenet_classification)}")
print(f"Sample prediction: {sample.lenet_classification}")
print(f"Prediction label: {sample.lenet_classification.label}")
print(f"Prediction confidence: {sample.lenet_classification.confidence}")
print(f"Prediction logits shape: {len(sample.lenet_classification.logits)}")

session.refresh()
print(f"\nFiftyOne App URL: {session.url}")



# ### Create a View Showing only LeNet's Mistakes
# 
# The FiftyOne `Classification` object allows to do filtering on the samples where we have issues.
# 

# In[ ]:


# Create a view showing only LeNet's misclassifications
misclassified_view = test_dataset.match(
    F("lenet_classification.label") != F("ground_truth.label")
)
print(f"Misclassified samples: {len(misclassified_view)} samples")


# ## Evaluating LeNet's Classification Performance
# 
# After applying our trained LeNet model to the test set, we need to evaluate its performance against the ground truth labels. This evaluation goes beyond simple accuracy to provide detailed insights into where and how the model succeeds or fails.
# FiftyOne's evaluation framework generates metrics including per-class precision, recall, and F1-scores, along with confusion matrices that reveal which digit pairs the model most often confuses. This analysis helps identify weaknesses and guides future improvements.
# 
# 

# In[ ]:


lenet_evaluation_results = test_dataset.evaluate_classifications(
    "lenet_classification",
    gt_field="ground_truth",
    eval_key="lenet_eval")

session.refresh()
print(session.url)


# In[ ]:


lenet_evaluation_results.print_report(digits=3)


# In[ ]:


lenet_evaluation_results.plot_confusion_matrix()


# In[ ]:


# Compute quantiles of confidence
test_dataset.quantiles("lenet_classification.confidence", [0.01, 0.05, 0.25, 0.5, 0.75] )


# In[ ]:


low_confidence_view = test_dataset.match(F("lenet_classification.confidence") < 0.69)
low_confidence_view


# In[ ]:


session.view = low_confidence_view
session.refresh()
print(session.url)


# ## Evaluating Sample Hardness and Mistakenness through Logits
# 
# **Quantifying Prediction Uncertainty and Label Quality**
# 
# Model logits contain rich information beyond simple predictions. By analyzing the raw output scores before softmax conversion (the logits), we can compute metrics that reveal which samples the model finds challenging and which labels may be questionable.
# 
# **Hardness** measures prediction uncertainty based on the model's confidence distribution. Samples with high hardness have flat probability distributions, indicating the model struggles to distinguish between classes. These often represent genuinely difficult cases or edge cases in the data.
# 
# **Mistakenness** identifies samples where the model's confident predictions disagree with ground truth labels. High mistakenness scores suggest potential annotation errors rather than model failures, as the model may have learned correct patterns that conflict with incorrect labels.
# 
# ```python
# # Compute hardness based on prediction uncertainty
# fob.compute_hardness(test_dataset, label_field='lenet_classification')
# 
# # Compute mistakenness to identify potential label errors  
# fob.compute_mistakenness(test_dataset,
#                         pred_field="lenet_classification",
#                         label_field="ground_truth")
# ```
# 
# These metrics transform raw model scores (logits) into actionable insights for the dataset

# In[ ]:


#Hardness is a measure computed based on model prediction output (through
#logits) that summarizes a measure of the uncertainty the model had with the
#sample. This makes hardness quantitative and can be used to detect things
#like hard samples and annotation errors
fob.compute_hardness(test_dataset,
                     label_field='lenet_classification',
                     )


# In[ ]:


# Evaluate sample mistakenness (how likely the sample is mislabeled)
# Samples with high mistakenness often have conflicting model output and ground truth
fob.compute_mistakenness(test_dataset,
                         pred_field="lenet_classification",
                         label_field="ground_truth")


# In[ ]:


# Compute quantiles of mistakenness
mistakenness_quantiles = test_dataset.quantiles(
    "mistakenness",
    [0.9, 0.95, 0.99]
)

print("\nMistakenness quantiles for train_dataset:")
print(mistakenness_quantiles)


# In[ ]:


suspicious_test_samples_view = test_dataset.match(
                             F("mistakenness") > mistakenness_quantiles[-1]
                             ).sort_by("mistakenness", reverse=True)


# In[ ]:


# Set the session view to one of the views to visualize it in the App
session.view = suspicious_test_samples_view
session.refresh()
print(f"\nFiftyOne App URL showing top mistakenness in the test set: {session.url}")


# ## Add Classifications to the Validation Dataset
# 
# We need to have Classification objects on the validation set to evaluate the quality of the validation dataset.

# In[ ]:


print("Applying best LeNet model to the validation set...")

# Set the loaded model to evaluation mode

# Lists to store predictions and logits
val_predictions = []
val_all_logits = []

# Run inference on the validation set
with torch.inference_mode(): # Disable gradient calculation
    for images, _ in tqdm(val_loader):
        images = images.to(device)

        # Forward pass to get logits
        logits = loaded_model(images)
        val_all_logits.append(logits.cpu().numpy()) # Store logits

        # Get predicted class indices
        _, predicted = torch.max(logits.data, 1)
        val_predictions.extend(predicted.cpu().numpy()) # Store predictions

# Concatenate logits from all batches
val_all_logits = np.concatenate(val_all_logits, axis=0)

print("Inference on validation set complete.")
print(f"Shape of collected validation logits: {val_all_logits.shape}")
print(f"Number of collected validation predictions: {len(val_predictions)}")

# Store the predictions and logits back into the FiftyOne dataset as Classification objects
print("Storing predictions and logits as FiftyOne Classifications for validation set...")

for i, sample in enumerate(val_dataset):
    # Get the predicted class index and corresponding class name
    predicted_idx = val_predictions[i]
    predicted_label = dataset_classes[predicted_idx]

    # Get logits for this sample
    sample_logits = val_all_logits[i]

    # Calculate confidence scores (softmax applied to logits)
    confidences = Fun.softmax(torch.tensor(sample_logits), dim=0).numpy()
    predicted_confidence = float(confidences[predicted_idx])

    # Create FiftyOne Classification object with prediction
    classification = fo.Classification(
        label=predicted_label,
        confidence=predicted_confidence,
        logits=sample_logits.tolist()  # Store raw logits
    )

    # Store the Classification object in the sample
    sample["lenet_validation_classification"] = classification

    sample.save()

print("Predictions and logits stored successfully as FiftyOne Classifications for validation set.")


# In[ ]:


# Evaluate sample hardness (how hard the sample is for the model)
# Hardness is a measure computed based on model prediction output (through
# logits) that summarizes a measure of the uncertainty the model had with the
# sample. This makes hardness quantitative and can be used to detect things
# like hard samples and annotation errors
fob.compute_hardness(val_dataset,
                     label_field='lenet_validation_classification',
                     )

# Evaluate sample mistakenness (how likely the sample is mislabeled)
# Samples with high mistakenness often have conflicting model output and ground truth
fob.compute_mistakenness(val_dataset,
                         pred_field="lenet_validation_classification",
                         label_field="ground_truth")

# Compute quantiles of mistakenness for the validation set
mistakenness_quantiles_val = val_dataset.quantiles(
    "mistakenness",
    [0.9, 0.95, 0.99]
)

print("\nMistakenness quantiles for validation_dataset:")
print(mistakenness_quantiles_val)

# Compute quantiles of hardness for the validation set
hardness_quantiles_val = val_dataset.quantiles(
    "hardness",
    [0.9, 0.95, 0.99]
)

print("\nHardness quantiles for validation_dataset:")
print(hardness_quantiles_val)



# In[ ]:


# Create a view of samples with high mistakenness in the validation set
suspicious_val_samples_view = val_dataset.match(
                             F("mistakenness") > mistakenness_quantiles_val[-1]
                             ).sort_by("mistakenness", reverse=True)

# Set the session view to one of the views to visualize it in the App
session.view = suspicious_val_samples_view
session.refresh()
print(f"\nFiftyOne App URL showing top mistakenness in the validation set: {session.url}")



# In[ ]:


bad_val_images_paths = set([
    "/root/fiftyone/mnist/train/data/005973.jpg",
    "/root/fiftyone/mnist/train/data/026623.jpg",
    "/root/fiftyone/mnist/train/data/025160.jpg",
    "/root/fiftyone/mnist/train/data/010383.jpg",
    "/root/fiftyone/mnist/train/data/031607.jpg",
    "/root/fiftyone/mnist/train/data/027027.jpg",
    "/root/fiftyone/mnist/train/data/013185.jpg",
    "/root/fiftyone/mnist/train/data/041291.jpg",
    "/root/fiftyone/mnist/train/data/032343.jpg",
    "/root/fiftyone/mnist/train/data/049961.jpg",
    "/root/fiftyone/mnist/train/data/031851.jpg",
    "/root/fiftyone/mnist/train/data/007348.jpg",
    "/root/fiftyone/mnist/train/data/035311.jpg",
    "/root/fiftyone/mnist/train/data/030417.jpg",
    "/root/fiftyone/mnist/train/data/008385.jpg",
    "/root/fiftyone/mnist/train/data/016749.jpg",
    "/root/fiftyone/mnist/train/data/031743.jpg",
    "/root/fiftyone/mnist/train/data/023869.jpg"
])


# In[ ]:


# Create a view of samples with high hardness in the validation set
hard_val_samples_view = val_dataset.match(
                         F("hardness") > hardness_quantiles_val[-1]
                         ).sort_by("hardness", reverse=True)

# Set the session view to visualize hard samples
session.view = hard_val_samples_view
session.refresh()
print(f"\nFiftyOne App URL showing top hardness in the validation set: {session.url}")


# ## Creating Embeddings from the LeNet-5 Model
# 
# **Extracting Internal Feature Representations**
# 
# Neural networks process information through multiple layers, with each layer transforming input data into increasingly abstract representations. While we typically focus on the final output layer for predictions, the intermediate layers contain valuable feature representations that reveal what the model has learned about the input data.
# 
# LeNet-5's architecture builds features hierarchically: early convolutional layers detect edges and simple patterns, while deeper layers combine these into more complex digit-specific features. The fully connected layers before the final classification layer contain rich embeddings that capture the essential characteristics the model uses to distinguish between digit classes.
# 
# By extracting these learned embeddings, we can analyze how our trained model represents each digit internally. Unlike pre-trained embeddings from models like CLIP, these features are optimized for our specific task of handwritten digit recognition. This makes them particularly valuable for understanding model behavior, identifying challenging samples, and comparing how different digits cluster in the learned feature space.
# 
# The extraction process uses PyTorch hooks to capture intermediate layer outputs during inference, allowing us to access the 84-dimensional feature vectors from the penultimate layer that encode each sample's representation in the model's learned space.
# 
# 
# 

# In[ ]:


## Extract Embeddings from LeNet Model Using PyTorch Hooks

def extract_lenet_embeddings(model, dataloader, device, layer_name='fc1'):
    """
    Extract embeddings from a specified layer of the LeNet model using PyTorch hooks.

    Args:
        model: Trained LeNet model
        dataloader: PyTorch DataLoader
        device: Device to run inference on
        layer_name: Name of the layer to extract embeddings from
                   Options: 'conv3', 'fc1', or 'fc2'

    Returns:
        numpy array of embeddings
    """
    # Dictionary to store the embeddings
    embeddings_dict = {}

    def hook_fn(module, input, output):
        """Hook function to capture layer outputs"""
        # Flatten the output if it's from conv layers
        if len(output.shape) > 2:
            embeddings_dict['embeddings'] = output.view(output.size(0), -1).cpu().detach()
        else:
            embeddings_dict['embeddings'] = output.cpu().detach()

    # Register the hook on the specified layer
    layer_map = {
        'conv3': model.conv3,  # Shape: (batch_size, 120, 1, 1) -> flattened to (batch_size, 120)
        'fc1': model.fc1,     # Shape: (batch_size, 84) - most common choice
        'fc2': model.fc2      # Shape: (batch_size, 10) - final logits
    }

    if layer_name not in layer_map:
        raise ValueError(f"Invalid layer_name. Choose from: {list(layer_map.keys())}")

    target_layer = layer_map[layer_name]
    hook_handle = target_layer.register_forward_hook(hook_fn)

    # Set model to evaluation mode
    model.eval()

    all_embeddings = []

    print(f"Extracting embeddings from {layer_name} layer...")

    with torch.inference_mode():
        for images, _ in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)

            # Forward pass (hook will capture the embeddings)
            _ = model(images)

            # Store the captured embeddings
            batch_embeddings = embeddings_dict['embeddings'].numpy()
            all_embeddings.append(batch_embeddings)

    # Remove the hook to clean up
    hook_handle.remove()

    # Concatenate all embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    print(f"Extracted embeddings shape: {final_embeddings.shape}")
    print(f"Embedding dimension: {final_embeddings.shape[1]}")

    return final_embeddings


# # Create a new DataLoader for the train_set  for inference
# 
# It's important to ensure that shuffling is disabled for this training data `DataLoader`, otherwise our predictions will not match the corresponding data points when we store them.
# 

# In[ ]:


# This uses the torch_train_set which is derived from train_dataset
train_inference_loader = torch.utils.data.DataLoader(
    torch_train_set,
    batch_size=batch_size,
    shuffle=False,    # CRITICAL: Must be False for ordered predictions
    num_workers=num_workers,
    pin_memory=True
)

# Set the loaded model to evaluation mode
loaded_model.eval()

# Lists to store predictions and logits for the training set
train_predictions = []
train_all_logits = []

# Run inference on the training set using the NON-SHUFFLED loader
print("Applying LeNet model to the train_dataset (using non-shuffled loader)...")
with torch.inference_mode(): # Disable gradient calculation
    # Use the new train_inference_loader
    for images, _ in tqdm(train_inference_loader, desc="Processing train batches for inference"):
        images = images.to(device)

        # Forward pass to get logits
        logits = loaded_model(images)
        train_all_logits.append(logits.cpu().numpy()) # Store logits

        # Get predicted class indices
        _, predicted = torch.max(logits.data, 1)
        train_predictions.extend(predicted.cpu().numpy()) # Store predictions

# Concatenate logits from all batches
train_all_logits = np.concatenate(train_all_logits, axis=0)

print("Inference on train_dataset complete.")
print(f"Shape of collected train logits: {train_all_logits.shape}")
print(f"Number of collected train predictions: {len(train_predictions)}")

# Store the predictions and logits back into the FiftyOne dataset as Classification objects
print("Storing predictions and logits as FiftyOne Classifications for train_dataset...")
for i, sample in enumerate(tqdm(train_dataset, desc="Storing train classifications")):
    predicted_idx = train_predictions[i]
    predicted_label = dataset_classes[predicted_idx] # Assuming dataset_classes is consistent
    sample_logits = train_all_logits[i]
    confidences = Fun.softmax(torch.tensor(sample_logits), dim=0).numpy()
    predicted_confidence = float(confidences[predicted_idx])
    classification = fo.Classification(
        label=predicted_label,
        confidence=predicted_confidence,
        logits=sample_logits.tolist()
    )
    sample["lenet_train_classification"] = classification
    sample.save()

print("Predictions and logits stored successfully as FiftyOne Classifications for train_dataset.")


# In[ ]:


lenet_train_evaluation_results = train_dataset.evaluate_classifications(
    "lenet_train_classification",
    gt_field="ground_truth",
    eval_key="lenet_train_eval"
)
session.refresh()
lenet_train_evaluation_results.print_report(digits=3)


# In[ ]:


session.view = train_dataset.view()
session.refresh()
print(session.url)


# ## Creating a View of the Mislabeled Images in the Training Data
# 
# **Identifying Potential Annotation Errors Through Model Predictions**
# 
# When a trained model disagrees with ground truth labels, this often reveals annotation errors rather than model failures. A well-trained neural network that has learned robust patterns from thousands of examples may be more reliable than human annotators on ambiguous cases. This principle becomes valuable for quality control in large datasets where manual verification of every sample is impractical.
# 
# **The Logic Behind Mislabel Detection**
# 
# Models learn statistical patterns across the entire dataset. When a model with high accuracy predicts a different label than the ground truth, several scenarios are possible: the model made an error, the annotation is incorrect, or the sample is ambiguous. By examining these disagreements, we can identify samples that warrant human review.
# 
# For MNIST, known annotation errors exist in the original dataset. Research has identified several hundred ambiguous or mislabeled samples where even human experts disagree on the correct digit. Our trained LeNet model, having learned from 50,000+ examples, may detect these problematic cases more than the original annotators.
# 
# Creating a filtered view of these disagreements allows us to focus human attention on the most questionable samples. This approach scales annotation quality control from reviewing entire datasets to examining only the cases where model and annotator disagree.
# 
# This approach to finding annotation errors represents a powerful application of model predictions for dataset improvement, transforming disagreements between model and labels into opportunities for enhanced data quality.
# 
# We focus on the mislabeled images from the training set, as these are the ones that we should first **augment** to improve the performance of the model.

# In[ ]:


# A view of mislabeled images from the training set
mislabeled_train_images_view = \
train_dataset.match(
    F("lenet_train_classification.label")!= F("ground_truth.label"))

mislabeled_train_images_view


# In[ ]:


session.view = mislabeled_train_images_view
session.refresh()
print(session.url)


# ## Extract Embeddings from the Trained LeNet
# 
# **Capturing Learned Feature Representations**
# 
# Neural networks learn hierarchical feature representations through their layers. While the final layer produces class predictions, intermediate layers contain rich embeddings that capture the visual patterns the model has learned to distinguish digits. These learned features often differ from pre-trained embeddings like CLIP, as they are optimized for the specific task of handwritten digit recognition.
# 
# Extracting embeddings from our trained LeNet model allows us to analyze what visual concepts the network has learned. The fully connected layer before the final classification layer (fc1) produces 84-dimensional vectors that represent each digit in the feature space that the model uses for decision-making.
# 
# 

# In[ ]:


print("Creating non-shuffled DataLoader for embedding extraction...")

# Create a DataLoader specifically for inference with NO SHUFFLING
train_inference_loader = torch.utils.data.DataLoader(
    torch_train_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

print(f"Non-shuffled inference DataLoader created with {len(train_inference_loader)} batches")

# Extract embeddings from the fc1 layer (84-dimensional representations)
print("Extracting LeNet embeddings with proper sample ordering...")

lenet_embeddings = extract_lenet_embeddings(
    model=loaded_model,
    dataloader=train_inference_loader,  # ✅ Use non-shuffled loader
    device=device,
    layer_name='fc1'  # 84-dimensional embeddings from fully connected layer
)


# In[ ]:


# Wrap embeddings to their associated filenames ()
embeddings_dict = {img_path: emb for img_path, emb in zip(torch_train_set.image_paths, lenet_embeddings)}


# ## Store LeNet Embeddings in FiftyOne Dataset
# 
# **Integrating Model Features with Dataset Metadata**
# 
# After extracting embeddings from our trained LeNet model, we need to store them in our FiftyOne dataset to enable analysis and visualization. This integration allows us to leverage FiftyOne's powerful querying and visualization capabilities on the learned feature representations.
# 
# Storing embeddings as sample fields transforms abstract neural network features into queryable fields. We can then compute similarity indices, create visualizations, and analyze how the model's learned representations relate to prediction accuracy and sample characteristics.

# In[ ]:


print("Storing LeNet embeddings in FiftyOne dataset...")

# Store embeddings in each sample
for index, sample in enumerate(tqdm(train_dataset, desc="Storing embeddings")):
    sample["lenet_embeddings"] = lenet_embeddings[index]
    sample.save()

print("LeNet embeddings stored successfully in samples from train_dataset.")

# Verify storage
sample = train_dataset.first()
print(f"Sample LeNet embedding shape: {sample.lenet_embeddings.shape}")
print(f"Embedding type: {type(sample.lenet_embeddings)}")

# Persisting the change
train_dataset.save()


# In[ ]:


train_dataset.first()


# In[ ]:


train_dataset.get_field_schema()


# ## Create a Similarity Index Based on the CNN embeddings
# 
# After extracting embeddings from our trained LeNet model, we create a **similarity index**. This index allows for efficient searching of samples based on their learned feature representations. Similar to the CLIP similarity index, this enables us to quickly find training images that are visually similar according to what the LeNet model has learned, which is useful for exploring clusters and identifying related samples in the model's feature space.
# 
# 
# 

# In[ ]:


print("Computing similarity index based on LeNet embeddings...")

lenet_similarity_index = fob.compute_similarity(
    train_dataset,  # Using the training dataset
    embeddings="lenet_embeddings",  # Field containing the LeNet embeddings
    brain_key="lenet_cosine_similarity_index",  # Unique identifier for this similarity index
    backend="sklearn",  # Can also use "pinecone" for large datasets
    metric="cosine"  # Similarity metric
)

print(f"LeNet similarity index computed successfully!")



# In[ ]:


# Display similarity index information
print(f"LeNet Similarity Index Details:")
print(f"- Total samples indexed: {len(train_dataset)}")
print(f"- Embeddings field: {lenet_similarity_index.config.embeddings_field}")
print(f"- Metric: {lenet_similarity_index.config.metric}")

# Example 1: Find similar images to a specific digit using LeNet embeddings
# Let's pick a sample from the training dataset
query_sample = train_dataset.first()
print(f"""\nQuerying for images similar to sample: {query_sample.id}
            with label {query_sample.ground_truth.label}
            using LeNet embeddings""")

# Use the sample ID (string) instead of the sample object
similar_view_lenet = train_dataset.sort_by_similarity(
    query_sample.id,  # Pass the sample ID
    brain_key="lenet_cosine_similarity_index",
    k=10,  # Return top 10 most similar
    reverse=False  # Most similar first
)

print(f"Found {len(similar_view_lenet)} most similar samples using LeNet embeddings")

# Update FiftyOne App view to show similar samples
session.view = similar_view_lenet
session.refresh()
print(session.url)


# ## Evaluating Image Representativeness and Uniqueness through their Embeddings
# 
# **Image representativeness** and **uniqueness** are metrics for understanding dataset quality and composition that work hand-in-hand with clustering analysis.
# 
# **Representativeness** measures how well a sample captures the central characteristics of its cluster, highly representative images sit near cluster centers and exemplify the common visual patterns within each group.
# 
# **Uniqueness** identifies samples that are distant from any cluster centers, often representing edge cases, rare scenarios, or potential annotation errors that clustering algorithms struggle to categorize.
# 
# These scores are normalized to [0, 1] and become particularly insightful when viewed alongside clustering results. Representative samples serve as ideal cluster exemplars for understanding what each group represents, while highly unique samples often fall between clusters or form singleton groups, indicating unusual data points that may require special handling in our machine learning pipeline.
# 

# In[ ]:


## Compute Uniqueness and Representativeness for LeNet Embeddings

print("Computing uniqueness scores based on LeNet embeddings...")
fob.compute_uniqueness(train_dataset, embeddings='lenet_embeddings')

print("Computing representativeness scores based on LeNet embeddings...")
fob.compute_representativeness(train_dataset, embeddings='lenet_embeddings')

print("Uniqueness and representativeness computation complete.")



# In[ ]:


# Compute mistakenness on samples on the train_set
# Mistakenness identifies samples where the model's confident predictions disagree with ground truth labels.
# High mistakenness scores suggest potential annotation errors rather than model failures.
fob.compute_mistakenness(train_dataset,
                         pred_field="lenet_train_classification",
                         label_field="ground_truth")

print("Mistakenness computed successfully for train_dataset.")



# In[ ]:


# Compute quantiles of mistakenness
mistakenness_quantiles = train_dataset.quantiles(
    "mistakenness",
    [0.9, 0.95, 0.99]
)

print("\nMistakenness quantiles for train_dataset:")
print(mistakenness_quantiles)


# In[ ]:


# Hardness is a measure computed based on model prediction output (through
# logits) that summarizes a measure of the uncertainty the model had with the
# sample. This makes hardness quantitative and can be used to detect things
# like hard samples and annotation errors
fob.compute_hardness(train_dataset,
                     label_field='lenet_train_classification',
                     )

print("Hardness computed successfully for train_dataset.")

session.refresh()
print(f"\nFiftyOne App URL: {session.url}")


# In[ ]:


# Compute quantiles of hardness
hardness_quantiles = train_dataset.quantiles(
    "hardness",
    [0.9, 0.95, 0.99]
)

print("\nHardness quantiles for train_dataset:")
print(hardness_quantiles)


# ## Create 2D Visualizations of LeNet Embeddings
# Reducing High-Dimensional Features to Visual Form
# Our LeNet embeddings exist in 84-dimensional space, making them impossible to visualize directly. Dimensionality reduction techniques transform these high-dimensional vectors into 2D points that preserve important relationships between samples.
# 
# Principal Component Analysis (PCA) finds the directions of maximum variance in the data and projects samples onto the two most important axes. This linear transformation preserves global structure and shows which samples vary most from the dataset center.
# 
# 
# UMAP (Uniform Manifold Approximation and Projection) uses non-linear techniques to preserve local neighborhoods while revealing cluster structure. UMAP often separates distinct groups more clearly than PCA, making it valuable for identifying how the model groups similar digits.
# 

# In[ ]:


print("Creating 2D visualizations of LeNet embeddings...")

# PCA visualization
pca_viz_lenet = fob.compute_visualization(
    train_dataset,
    method="pca",
    embeddings="lenet_embeddings",
    num_dims=2,
    brain_key="pca_lenet_embeddings"
)

# UMAP visualization
umap_viz_lenet = fob.compute_visualization(
    train_dataset,
    method="umap",
    embeddings="lenet_embeddings",
    num_dims=2,
    brain_key="umap_lenet_embeddings"
)

print("2D visualizations created successfully.")


# #### PCA 2D Projection of the CNN Embedding Space
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/pca_lenet_embeddings.png?raw=true)
# 
# 
# #### UMAP 2D Projection of the CNN Embedding Space
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/umap_lenet_embeddings.png?raw=true)

# Inspect the projected embeddings on the FiftyOne app. Compare how the PCA and the UMAP projections look like.

# In[ ]:


session.refresh()
print(session.url)


# ## Exploring Uniqueness and Representativeness with Dataset Aggregations
# 
# **Understanding Score Distributions Through Statistical Analysis**
# 
# We can use FiftyOne's aggregation methods to explore the distribution of uniqueness and representativeness values across our dataset. This analysis helps us identify meaningful thresholds for categorizing our samples based on their statistical properties.
# 
# **Quantile Analysis for Threshold Selection**
# 
# Quantiles divide our data into meaningful segments, revealing the distribution shape and enabling principled threshold selection. The 25th, 50th (median), 75th, and 95th percentiles help us understand where most samples fall and identify the truly exceptional cases.
# 
# For uniqueness scores, the 95th percentile reveals the most unusual samples that fall far from any cluster centers. These often represent edge cases, annotation errors, or rare variants that deserve special attention. The 75th percentile for representativeness identifies samples that exemplify their clusters well, making them ideal candidates for visualization or as training exemplars.
# 
# ```python
# # Get quantiles for uniqueness scores
# uniqueness_quantiles = train_dataset.aggregate(fo.Quantiles("uniqueness",
#                                           [0.05, 0.25, 0.5, 0.75, 0.95]))
# 
# # Get quantiles for representativeness scores  
# representativeness_quantiles = train_dataset.aggregate(fo.Quantiles("representativeness",
#                                     [0.05, 0.25, 0.5, 0.75, 0.95]))
# ```
# 
# These quantile boundaries enable us to create filtered views targeting specific sample types, such as the top 5% most unique samples or the most representative examples from each cluster.

# In[ ]:


# Get quantiles for uniqueness based on LeNet embeddings

quantiles_of_interest = [0.5, 0.75, 0.95, 0.99, 0.999]
uniqueness_quantiles = train_dataset.quantiles("uniqueness", quantiles_of_interest)
quantile_labels = [f"{label*100:.1f}% " for label in quantiles_of_interest]
print('Uniqueness quantiles')
{label: value for label, value in zip(quantile_labels, uniqueness_quantiles)}


# In[ ]:


quantiles_of_interest = [0.5, 0.75, 0.95, 0.99, 0.999]
representativeness_quantiles = train_dataset.quantiles("uniqueness", quantiles_of_interest)
print('Representativeness quantiles')
{label: value for label, value in zip(quantile_labels, representativeness_quantiles)}


# ## Create Views for Most Unique and Most Representative Samples
# 
# We can use filtered Views of the dataset to get the most unique values and launch the FiftyOne app from them.

# In[ ]:


# Most unique samples (top 0.1% - these are often edge cases or outliers)
most_unique_samples_view = train_dataset.match(
                             F("uniqueness") > uniqueness_quantiles[-1]
                             ).sort_by("uniqueness", reverse=True)

print(f"Most unique samples: {len(most_unique_samples_view)} samples")

# Most representative samples (top 0.1% - these exemplify their clusters well)
most_representative_samples_view = train_dataset.match(
    F("representativeness") > representativeness_quantiles[-1]
    ).sort_by("representativeness", reverse=True)

print(f"Most representative samples: {len(most_representative_samples_view)} samples")


# In[ ]:


## Visualize most unique images
session.view = most_unique_samples_view
session.refresh()
print(f"FiftyOne App URL: {session.url}")


# In[ ]:


# Get the sample with the highest uniqueness score
most_unique_sample = most_unique_samples_view.first()

print(f"""Most unique sample ID: {most_unique_sample.id}
Uniqueness score: {most_unique_sample.uniqueness:.4f}
Ground truth label: {most_unique_sample.ground_truth.label}""")

# Find images most similar to the most unique one using the LeNet similarity index
# We want the most similar samples, so reverse=False
similar_to_unique_view = train_dataset.sort_by_similarity(
    most_unique_sample.id,
    brain_key="lenet_cosine_similarity_index", # Use the LeNet index
    k=20, # Get the top 20 similar samples (including the unique one itself)
    reverse=False
)

print(f"Found {len(similar_to_unique_view)} samples most similar to the most unique one.")

# Visualize the samples similar to the most unique one
session.view = similar_to_unique_view
session.refresh()
print(session.url)


# In[ ]:


## Visualize most representative images
session.view = most_representative_samples_view
session.refresh()
print(f"FiftyOne App URL: {session.url}")


# ## Obtain PyTorch Datasets from the Most Unique and Representative Samples
# 
# We can create new PyTorch datasets from the samples that are most unique or representative and use them to retrain the model.  

# In[ ]:


most_unique_torch_train_ds = CustomTorchImageDataset(most_unique_samples_view,
                                                     label_map=label_map,
                                                     image_transforms=image_transforms)
most_representative_torch_train_ds = CustomTorchImageDataset(most_representative_samples_view,
                                                             label_map=label_map,
                                                             image_transforms=image_transforms)


# In[ ]:


new_train_set = ConcatDataset([most_unique_torch_train_ds, most_representative_torch_train_ds])


# **Suggested Exercise**
# Create a new DataLoader and retrain the model using **only** the most representative and unique samples. On the next sections there is code that can be used as guide.

# ## Augmenting the Misclassified Training Samples and Retrain
# 
# 

# ### Effective Augmentations for MNIST
# 
# 
# ### Geometric Transformations:
# 
# * Small rotations (±10-15 degrees): Handwritten digits naturally vary in orientation
# * Small translations (±2-3 pixels): Accounts for centering variations in digit positioning
# * Slight scaling (0.9-1.1x): Handles size variations in handwriting
# * Moderate elastic deformations are  useful for MNIST because they simulate the natural variations in handwriting style. Imagine stretching and compressing parts of a digit as different people might write them.
# ### Why These Work
# The principle behind effective augmentation is creating realistic variations that preserve the digit's identity while exposing the model to plausible distortions. MNIST digits are centered and normalized, so augmentations should introduce controlled variability without making digits unrecognizable.
# ### Augmentations to Avoid
# * Heavy distortions like large rotations (>20°), extreme scaling, or aggressive elastic deformation can make digits ambiguous, a rotated "6" might look like a "9", or a heavily stretched "1" might resemble a "7".
# 
# * Color-based augmentations (brightness, contrast) have limited benefit since MNIST is grayscale and already normalized.

# ## Retraining and Final Evaluation
# 
# **Measuring the Impact of Augmented Data**
# 
# After identifying misclassified training samples and creating augmented versions, we retrain our model to measure performance improvements. This process demonstrates how targeted data augmentation can address specific model weaknesses while maintaining overall performance.
# 
# The retraining phase uses the original best model weights as a starting point, then fine-tunes on the combined dataset containing both original training data and augmented versions of problematic samples. We use a lower learning rate to preserve learned features while adapting to the new data.
# 
# Final evaluation compares the retrained model against both the original LeNet performance and CLIP's zero-shot results. This comparison reveals the relative benefits of supervised learning with augmentation versus pre-trained model capabilities.
# 
# ```python
# # Load best model for retraining
# retrain_model = ModernLeNet5()
# retrain_model.load_state_dict(torch.load(best_model_path, map_location=device))
# 
# # Retrain with combined dataset
# combined_train_dataset = ConcatDataset([torch_train_set, torch_augmented_dataset])
# ```
# 
# The evaluation metrics quantify whether targeted augmentation successfully improved model robustness and generalization on the held-out test set.

# In[ ]:


## Data Augmentation for Misclassified Training Samples

print(f"Number of misclassified training samples: {len(mislabeled_train_images_view)}")



# 
# ## Samples to Filter-out During Fine-tuning
# 
# 

# In[ ]:


# Identify samples in the top 0.1 percentile of mistakenness (likely annotation errors)mistakenness_99th_percentile = train_dataset.quantiles("mistakenness", [0.99])[0]
mistakenness_99th_percentile = train_dataset.quantiles("mistakenness", [0.999])[0]
highly_mistaken_view = train_dataset.match(F("mistakenness") > mistakenness_99th_percentile)

# Identify samples in the top 1 percentile of hardness (likely genuinely hard samples)
hardness_99th_percentile = train_dataset.quantiles("hardness", [0.999])[0]
highly_hard_view = train_dataset.match(F("hardness") > hardness_99th_percentile)

print(f"Number of highly mistaken samples (top 1%): {len(highly_mistaken_view)}")
print(f"Number of highly hard samples (top 1%): {len(highly_hard_view)}")

# Get the IDs of samples that are highly mistaken or highly hard
highly_mistaken_ids = set(highly_mistaken_view.values("id"))
highly_hard_ids = set(highly_hard_view.values("id"))

# Combine the sets of IDs that are hard
difficult_samples = highly_mistaken_ids.union(highly_hard_ids)

print(f"Total number of difficult samples (union of highly mistaken and highly hard): {len(difficult_samples)}")



# In[ ]:


# Get the IDs of the misclassified samples
misclassified_sample_ids = set(mislabeled_train_images_view.values("id"))

print(f"Total number of misclassified samples in the training set: {len(misclassified_sample_ids)}")

# Find the intersection of difficult samples and misclassified samples
intersection_difficult_misclassified_ids = difficult_samples.intersection(misclassified_sample_ids)

print(f"Number of samples that are both difficult and misclassified: {len(intersection_difficult_misclassified_ids)}")

# Create a FiftyOne view containing only these intersection samples
intersection_difficult_misclassified_view = train_dataset.select(
    list(intersection_difficult_misclassified_ids)
    ).sort_by('hardness',
              reverse=True)


print(f"Created a FiftyOne view for the intersection samples with {len(intersection_difficult_misclassified_view)} samples.")


# Inspect the samples in ids_to_remove, we can decide whether
# whether we augment them or exclude them from the fine-tuning data
session.view = intersection_difficult_misclassified_view
session.refresh()
print(f"\nFiftyOne App URL: {session.url}")


# ### Finding the Needles in the Haystack
# 
# ![](https://raw.githubusercontent.com/andandandand/practical-computer-vision/refs/heads/main/images/mistaken_images.png)
# 
# Using our metrics we have reduced significantly the search space and the manual inspection for our problematic samples. Isn't it interesting to find these samples in one of the most canonical datasets in machine learning?

# In[ ]:


filenames_of_samples_to_exclude = set([
                                      "/root/fiftyone/mnist/train/data/002902.jpg",
                                      "/root/fiftyone/mnist/train/data/003693.jpg",
                                      "/root/fiftyone/mnist/train/data/035247.jpg",
                                      "/root/fiftyone/mnist/train/data/018383.jpg",
                                      "/root/fiftyone/mnist/train/data/033507.jpg",
                                      "/root/fiftyone/mnist/train/data/039424.jpg",
                                      "/root/fiftyone/mnist/train/data/025547.jpg",
                                      "/root/fiftyone/mnist/train/data/047218.jpg",
                                      "/root/fiftyone/mnist/train/data/001921.jpg",
                                      "/root/fiftyone/mnist/train/data/004477.jpg",
                                      "/root/fiftyone/mnist/train/data/050606.jpg",
                                      "/root/fiftyone/mnist/train/data/030050.jpg",
                                      "/root/fiftyone/mnist/train/data/024614.jpg",
                                      "/root/fiftyone/mnist/train/data/014583.jpg",
                                      "/root/fiftyone/mnist/train/data/034521.jpg",
                                      "/root/fiftyone/mnist/train/data/050995.jpg",
                                      "/root/fiftyone/mnist/train/data/004477.jpg",
                                      "/root/fiftyone/mnist/train/data/033319.jpg",
                                      "/root/fiftyone/mnist/train/data/034405.jpg"
                                      "/root/fiftyone/mnist/train/data/007260.jpg",
                                      "/root/fiftyone/mnist/train/data/001401.jpg",
                                      "/root/fiftyone/mnist/train/data/002149.jpg",
                                      "/root/fiftyone/mnist/train/data/000903.jpg",
                                      "/root/fiftyone/mnist/train/data/001921.jpg"
                                      ])

# Use match() to filter by filepath
samples_to_exclude_view = train_dataset.match(
    fo.ViewField("filepath").is_in(filenames_of_samples_to_exclude)
)
print('Samples to exclude from new training runs')
session.view = samples_to_exclude_view
session.refresh()
print(session.url)


# In[ ]:


# get the ids from samples_to_exclude_view
ids_of_samples_to_exclude = samples_to_exclude_view.values("id")
ids_of_samples_to_exclude


# ## Create a New Training Dataset without the Excluded Samples

# In[ ]:


new_train_dataset = train_dataset.exclude(ids_of_samples_to_exclude).clone()

print(f"Original train_dataset has {len(train_dataset)} samples.")
print(f"Excluding {len(new_train_set)} samples.")
print(f"New train_dataset view has {len(new_train_set)} samples.")

new_train_dataset.name = "cleaned-up-mnist-training-set"
new_train_dataset.persistent = True
print(f"New persistent dataset created with {len(new_train_dataset)} samples.")



# In[ ]:


# To work with this view in PyTorch, create a CustomTorchImageDataset from it
new_torch_train_set = CustomTorchImageDataset(
    new_train_dataset,  # Changed from new_train_set to new_train_dataset
    label_map=label_map, # Use the same label map
    image_transforms=image_transforms # Use the same transforms
)

print(f"New PyTorch dataset created from the filtered view with {len(new_torch_train_set)} samples.")

# You can now use new_torch_train_set to create a new DataLoader for retraining
# new_train_loader = create_deterministic_training_dataloader(
#     new_torch_train_set,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     pin_memory=True
# )
# print(f"New train DataLoader created with {len(new_train_loader)} batches.")


# **Another possibility:** using these samples to train a model with an 'IDK' (I don't know class).
# 
# Also consider that the validation and test set might have dirty samples too. Ideally we should repeat this process with them too.  

# ## Define Augmentations
# 
# **Configuring Transformations for MNIST Digit Recognition**
# 
# Data augmentation applies controlled transformations to training images, creating variations that help models generalize to new handwriting styles. For MNIST, effective augmentations simulate natural variations in digit writing without changing the digit's identity.
# 
# **Seeding for Reproducibility**: Set random seeds before defining augmentations to ensure consistent results across training runs. This enables reproducible experiments and fair model comparisons.
# 
# **MNIST-Specific Transformations**: Small rotations, translations, and elastic deformations work well for handwritten digits. These transformations mimic natural handwriting variations while preserving digit readability. Avoid extreme distortions that could make a "6" look like a "9" or render digits unrecognizable.
# 
# **Exploration**:
# To inspect the effect of the defined augmentations on the images, you can try [using the FiftyOne plugin for this](https://github.com/jacobmarks/fiftyone-albumentations-plugin).
# 

# In[ ]:


# Install the augmentations plugin
get_ipython().system('fiftyone plugins download https://github.com/jacobmarks/fiftyone-albumentations-plugin')


# In[ ]:


mnist_augmentations_1 = A.Compose([

    # Use Affine transform for shifting, scaling, and rotating
    A.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # ±10% translation
        scale=(0.9, 1.1),     # ±10% scaling
        rotate=(-5, 5),     # ±5° rotation
        p=0.8
    ),

    # Elastic deformations to simulate handwriting style variations
    A.ElasticTransform(
        alpha=20,             # Strength of distortion
        sigma=5,              # Smoothness of distortion
        border_mode=cv2.BORDER_CONSTANT,
        p=0.6
    ),

    # Mild perspective transformations
    A.Perspective(scale=(0.01, 0.03), p=0.2),

    # Mild grid distortion
    A.GridDistortion(num_steps=2, distort_limit=0.05, p=0.2),

])


# In[ ]:


# Define second set of augmentations
mnist_augmentations_2 = A.Compose([

    # Equalize the image histogram (can help with contrast variations)
    A.Equalize(p=0.3),

    # Affine with shear (simulates slanted handwriting)
    A.Affine(
        shear=(-5, 5),  # Add shear
        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)}, # Smaller translation
        scale=(0.98, 1.02), # Smaller scaling
        rotate=(-2, 2), # Smaller rotation
        border_mode=cv2.BORDER_CONSTANT,
        p=0.9
    ),

    # Mild perspective transformations
    A.Perspective(scale=(0.01, 0.03), p=0.2),

    # Mild grid distortion
    A.GridDistortion(num_steps=2, distort_limit=0.05, p=0.2),
])


# In[ ]:


# Define a third set of augmentations for MNIST
mnist_augmentations_3 = A.Compose([
    # Use Affine transform for shifting, scaling, and rotating
    A.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # ±10% translation
        scale=(0.95, 1.15),     # ±15% scaling
        rotate=(-7, 7),         # ±7° rotation
        p=0.8
    ),

    # Elastic deformation with different parameters
    A.ElasticTransform(
        alpha=15,
        sigma=4,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.4
    ),

    # Coarse Dropout (simulates occlusions or pen lifts)
    A.CoarseDropout(
        p=0.3
    ),
])



# ## Create the torch Dataset for the Augmented Set
# 
# **Building a Custom PyTorch Dataset for Augmented Training Data**
# 
# Creating a PyTorch Dataset for augmented data requires handling multiple variations of problematic samples while maintaining compatibility with PyTorch's DataLoader system. Our `AugmentedMNISTDataset` class applies augmentations during training, ensuring each epoch sees different variations of misclassified samples.
# 
# **Key Design Principles:**
# 
# **Dynamic Augmentation**: Apply transformations during data loading rather than pre-generating samples. This saves disk space while providing variation potential.
# 
# **Deterministic Access**: Use mathematical mapping for consistent sample ordering while allowing multiple augmented versions of each base sample. The formula `base_idx = idx // (augment_factor + 1)` determines which original sample to use.
# 
# **Integration**: Inherit from PyTorch's `Dataset` interface for compatibility with DataLoader features like batching, shuffling, and multi-process loading.
# 
# **Memory and Performance Considerations:**
# 
# **Memory Usage**: Store file paths rather than loaded images to maintain small memory footprint regardless of augmentation factor.
# 
# **Caching Strategy**: Consider implementing an LRU cache to store recent base images, reducing disk I/O during training.
# 
# **Augmentation Randomness**: Each call to `__getitem__` may produce different results due to random augmentation parameters.
# 
# **Error Resilience**: Robust error handling ensures training continues even if individual samples fail to load or augment.
# 
# This design provides a foundation for targeted data augmentation while maintaining flexibility to experiment with different augmentation strategies.

# In[ ]:


class AugmentedMNISTDataset(Dataset):
    """
    PyTorch dataset that applies augmentations to misclassified MNIST samples.
    Each sample can be augmented multiple times to create more training data.
    """

    def __init__(self, fiftyone_view,
                 label_map,
                 base_transforms,
                 augmentations=None,
                 augment_factor=5):
        """
        Args:
            fiftyone_view: FiftyOne view of misclassified samples
            label_map: Mapping from string labels to indices
            base_transforms: Base PyTorch transforms (normalization, etc.)
            augmentations: Albumentations transform pipeline
            augment_factor: How many augmented versions to create per sample
        """
        self.image_paths = fiftyone_view.values("filepath")
        self.str_labels = fiftyone_view.values("ground_truth.label")
        self.label_map = label_map
        self.base_transforms = base_transforms
        self.augmentations = augmentations
        self.augment_factor = augment_factor

        print(f"AugmentedMNISTDataset: {len(self.image_paths)} base samples")
        print(f"With augmentation factor {augment_factor}: {len(self)} total samples")

    def __len__(self):
        return len(self.image_paths) * (self.augment_factor + 1)  # +1 for original

    def __getitem__(self, idx):
        # Determine which base sample and whether to augment
        base_idx = idx // (self.augment_factor + 1)
        aug_idx = idx % (self.augment_factor + 1)

        # Load image
        image_path = self.image_paths[base_idx]
        image = Image.open(image_path).convert('L')

        # Convert to numpy for albumentations
        image_np = np.array(image, dtype=np.uint8)

        # Apply augmentation if not the first version (original)
        if aug_idx > 0 and self.augmentations is not None:
            augmented = self.augmentations(image=image_np)
            image_np = augmented['image']

        # Convert back to PIL for PyTorch transforms
        image = Image.fromarray(image_np, mode='L')

        # Apply base transforms (normalization, tensor conversion)
        if self.base_transforms:
            image = self.base_transforms(image)

        # Get label
        label_str = self.str_labels[base_idx]
        label_idx = self.label_map.get(label_str, -1)

        return image, torch.tensor(label_idx, dtype=torch.long)



# In[ ]:


# Calculate the 1th percentile (lowest percentile) of confidence scores for lenet_train_classification
confidence_quantiles = train_dataset.quantiles("lenet_train_classification.confidence", [0.01])
q10_confidence = confidence_quantiles[0]

print(f"The 1th percentile of confidence is: {q10_confidence:.4f}")

# Create a view of samples that were correctly classified AND have confidence <= 1th percentile
low_confidence_correct_view = train_dataset.match(
    (F("lenet_train_classification.label") == F("ground_truth.label")) &
    (F("lenet_train_classification.confidence") <= q10_confidence)
).sort_by("lenet_train_classification.confidence")

print(f"\nNumber of correctly classified samples with confidence <= {q10_confidence:.4f}: {len(low_confidence_correct_view)}")

# We can launch the FiftyOne App to visualize this view
session.view = low_confidence_correct_view
session.refresh()
print(session.url)


# ## Define Samples to Augment
# 
# This section focuses on identifying specific subsets of samples from the training data that are most likely to benefit from augmentation. We will select samples that were previously misclassified, are highly unique or representative in the embedding space, or were correctly classified but with low confidence. Augmenting these samples aims to improve the model's performance on challenging or underrepresented examples.
# 
# 
# 

# In[ ]:


dataset_name = "digit-samples-to-augment"

if dataset_name in fo.list_datasets():
    print(f"Dataset '{dataset_name}' already exists. Deleting it.")
    fo.delete_dataset(dataset_name)

# Always create a new dataset after deleting or if it didn't exist
dataset_to_augment = fo.Dataset(dataset_name)

for sample in mislabeled_train_images_view:
    dataset_to_augment.add_sample(sample)

for sample in most_unique_samples_view:
    dataset_to_augment.add_sample(sample)

for sample in most_representative_samples_view:
    dataset_to_augment.add_sample(sample)

for sample in low_confidence_correct_view:
    dataset_to_augment.add_sample(sample)

dataset_to_augment.save()


# In[ ]:


# Filter out excluded samples
dataset_to_augment = dataset_to_augment.exclude(ids_of_samples_to_exclude).clone()


# In[ ]:


torch_augmented_dataset = AugmentedMNISTDataset(
    dataset_to_augment,
    label_map=label_map,
    base_transforms=image_transforms,
    augmentations=mnist_augmentations_1, # We can also try mnist_augmentations_2 or mnist_augmentations_3
    augment_factor=9  # Create this number of augmented versions per input sample
)

print(f"Original misclassified samples: {len(mislabeled_train_images_view)}")
print(f'')
print(f"Total augmented dataset size: {len(torch_augmented_dataset)}")



# In[ ]:


# Combine the original dataset with its augmentation
combined_dataset = ConcatDataset([new_torch_train_set, torch_augmented_dataset])
print(f"Combined dataset size: {len(combined_dataset)}")


# In[ ]:


# Create new DataLoader for combined dataset
combined_train_loader = create_deterministic_training_dataloader(
    combined_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)


print(f"Combined DataLoader has {len(combined_train_loader)} batches.")


# ## Visualize Sample Augmentations in FiftyOne
# 
# **Inspecting Augmentation Quality Before Training**
# 
# Visualizing augmented samples ensures transformations improve model training rather than corrupt the data. FiftyOne provides an interface to compare original images with their augmented variants, helping validate that augmentations preserve digit identity while adding useful variation.
# 
# **Quality Control**: Review augmented samples to confirm that rotations, translations, and elastic deformations remain within acceptable bounds. A properly augmented "3" should still look like a "3" despite the applied transformations.
# 
# **Parameter Tuning**: Visual inspection helps adjust augmentation parameters. If elastic deformations make digits unrecognizable or rotations create ambiguous orientations, reduce the transformation intensity before training.
# 
# **Dataset Creation**: The visualization process creates temporary augmented samples that can be loaded into FiftyOne for side-by-side comparison with originals, enabling systematic evaluation of augmentation effectiveness.

# In[ ]:


# Create a small dataset with original and augmented versions for visualization
def create_augmentation_samples(view, augmentations, num_samples=50):
    """Create FiftyOne samples showing original and augmented versions"""

    dataset_name = "augmented_images_training_set"

    if dataset_name in fo.list_datasets():
        aug_viz_dataset = fo.load_dataset(dataset_name)
    else:
        aug_viz_dataset = fo.Dataset(dataset_name)

    sample_paths = view.values("filepath")[:num_samples]
    sample_labels = view.values("ground_truth.label")[:num_samples]

    for i, (path, label) in enumerate(zip(sample_paths, sample_labels)):
        # Load original image
        original_image = Image.open(path).convert('L')
        original_np = np.array(original_image)

        # Create sample for original image
        original_sample = fo.Sample(filepath=path)
        original_sample.tags = ["original"]
        original_sample["ground_truth"] = fo.Classification(label=label)
        original_sample["augmentation_type"] = "original"
        aug_viz_dataset.add_sample(original_sample)

        # Create 3 augmented versions
        for aug_idx in range(3):
            # Apply augmentation
            augmented = augmentations(image=original_np)['image']

            # Save augmented image temporarily
            aug_image = Image.fromarray(augmented, mode='L')
            temp_path = f"/tmp/aug_{i}_{aug_idx}.png"
            aug_image.save(temp_path)

            # Create FiftyOne sample for augmented image
            aug_sample = fo.Sample(filepath=temp_path)
            aug_sample.tags = ["augmented"]
            aug_sample["ground_truth"] = fo.Classification(label=label)
            aug_sample["augmentation_type"] = f"augmented_{aug_idx + 1}"
            aug_sample["original_sample_id"] = str(i)

            aug_viz_dataset.add_sample(aug_sample)

    return aug_viz_dataset

print("Creating augmentation visualization dataset...")

# Create the visualization dataset
aug_viz_dataset = create_augmentation_samples(
    dataset_to_augment,
    mnist_augmentations_1,
    num_samples=50
)

print(f"Created visualization dataset with {len(aug_viz_dataset)} samples")
print(f"Original samples: {len(aug_viz_dataset.match_tags('original'))}")
print(f"Augmented samples: {len(aug_viz_dataset.match_tags('augmented'))}")

# Launch FiftyOne App to visualize the augmentations
session.view = aug_viz_dataset.view()

print(f"\nAugmentation Visualization URL: {session.url}")


# In[ ]:


# Create views for easy comparison
original_view = aug_viz_dataset.match_tags("original")
augmented_view = aug_viz_dataset.match_tags("augmented")

print(f"\nTo compare:")
print(f"- View original samples: session.view = original_view")
print(f"- View augmented samples: session.view = augmented_view")
print(f"- Group by original sample: Use 'original_sample_id' field to group related samples")

# Add some helpful aggregations
print(f"\nSample distribution:")
print(f"By augmentation type: {aug_viz_dataset.count_values('augmentation_type')}")
print(f"By ground truth label: {aug_viz_dataset.count_values('ground_truth.label')}")


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/augmentation_mnist_vis.png?raw=true)

# ## Fine-Tuning with Augmented Data
# 
# The retraining process combines our original training data with the augmented versions of misclassified samples, allowing the model to learn from its previous mistakes. We use a lower learning rate to preserve the knowledge already learned while adapting to the new augmented examples.
# 
# During retraining, we monitor both training and validation loss to ensure the model improves without overfitting. The validation set continues to serve as our guide for saving the best model weights, ensuring we capture improvements in generalization rather than just memorization of the augmented data.
# 
# ```python
# for epoch in range(retrain_epochs):
#     train_loss = train_epoch(retrain_model, combined_train_loader)
#     val_loss = val_epoch(retrain_model, val_loader)
#     
#     if val_loss < best_retrain_val_loss:
#         torch.save(retrain_model.state_dict(), retrain_model_save_path)
# ```
# 
# This targeted approach allows us to address specific model weaknesses while maintaining overall performance on the broader dataset.

# In[ ]:


## Load Best Model for Retraining

# Ensure reproducibility for the retraining process
set_seeds(51) # Use the same seed as initial training for consistency

# Load the previously saved best model
best_model_path = Path(os.getcwd()) / 'best_lenet.pth'
retrain_model = ModernLeNet5()
retrain_model.load_state_dict(torch.load(best_model_path, map_location=device))
retrain_model = retrain_model.to(device)

print(f"Loaded best model from {best_model_path} for retraining")

# Use a lower learning rate for fine-tuning
retrain_optimizer = Adam(retrain_model.parameters(), lr=0.0001,
                       weight_decay=1e-3)
# 10x stronger regularization
# 30x smaller learning rate

print("\nStarting retraining with augmented data...")

retrain_epochs = 15
retrain_losses = []
retrain_val_losses = []

# Track the best validation loss during retraining
best_retrain_val_loss = float('inf')
retrain_model_save_path = Path(os.getcwd()) / 'retrained_lenet.pth'

for epoch in range(retrain_epochs):
    print(f"\n--- Retrain Epoch {epoch+1}/{retrain_epochs} ---")

    # Training phase
    retrain_model.train()
    batch_losses = []

    for images, labels in tqdm(combined_train_loader, desc="Retraining"):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = retrain_model(images)
        loss_value = ce_loss(logits, labels)

        # Backward pass
        retrain_optimizer.zero_grad()
        loss_value.backward()
        retrain_optimizer.step()

        batch_losses.append(loss_value.item())

    train_loss = np.mean(batch_losses)
    retrain_losses.append(train_loss)

    # Validation phase
    retrain_model.eval()
    val_batch_losses = []

    with torch.inference_mode():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            logits = retrain_model(images)
            loss_value = ce_loss(logits, labels)
            val_batch_losses.append(loss_value.item())

    val_loss = np.mean(val_batch_losses)
    retrain_val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save best model during retraining
    if val_loss < best_retrain_val_loss:
        best_retrain_val_loss = val_loss
        torch.save(retrain_model.state_dict(), retrain_model_save_path)
        print("✓ Saved improved retrained model")

print(f"\nRetraining complete! Best model saved to {retrain_model_save_path}")




# ## Plot Re-Training Progress
# 
# **Visualizing Training Dynamics Across Phases**
# 
# Training visualization reveals model learning patterns during both initial training and retraining with augmented data. Loss curves show convergence behavior, overfitting signals, and the impact of targeted data augmentation.
# 
# **Two-Panel Analysis**
# 
# The left panel shows retraining progress in isolation, tracking how the model adapts to augmented misclassified training samples.
# 
# The right panel provides historical context by plotting the complete training timeline. The vertical line marks where retraining begins, allowing comparison between original learning dynamics and fine-tuning behavior.
# 
# **Key Patterns to Observe**
# 
# - **Convergence Speed**: Retraining typically converges faster than initial training due to pre-learned features
# - **Loss Magnitude**: Validation loss during retraining should remain close to or below original best values  
# - **Stability**: Smooth loss curves indicate stable learning, while oscillations suggest learning rate issues
# - **Gap Analysis**: Small train-validation gaps indicate good generalization
# 
# **Interpreting Results**
# 
# Successful retraining shows decreasing validation loss without diverging from training loss. If validation loss increases while training loss decreases, we have trained the model enough. We will be using the state of the model that achieved the lowest validation loss.
# 

# In[ ]:


plt.figure(figsize=(12, 5))

# Plot retraining losses
plt.subplot(1, 2, 1)
plt.plot(retrain_losses, label='Retrain - Training Loss', marker='o')
plt.plot(retrain_val_losses, label='Retrain - Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Retraining Progress ({retrain_epochs} Epochs)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot comparison with original training (if we have the data)
plt.subplot(1, 2, 2)
if 'train_losses' in locals() and 'val_losses' in locals():
    epochs_orig = range(1, len(train_losses) + 1)
    epochs_retrain = range(len(train_losses) + 1, len(train_losses) + 1 + len(retrain_losses))

    plt.plot(epochs_orig, train_losses, label='Original Train', alpha=0.7)
    plt.plot(epochs_orig, val_losses, label='Original Val', alpha=0.7)
    plt.plot(epochs_retrain, retrain_losses, label='Retrain Train', marker='o', linewidth=2)
    plt.plot(epochs_retrain, retrain_val_losses, label='Retrain Val', marker='s', linewidth=2)

    plt.axvline(x=len(train_losses), color='red', linestyle='--', alpha=0.5, label='Retrain Start')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Complete Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ## Applying the Fine-tuned Model to the Test Set
# 
# Following is a brief description of what we are doing in this section.
# 
# **Evaluating Augmented Model Performance**
# 
# * Load the best retrained model weights and apply to the test dataset.
# 
# * Generate predictions and logits for performance comparison against the original model.
# 
# **Model Loading Process**
# 
# * Instantiate a fresh model architecture and load the saved state dictionary from retraining.
# 
# * Move to the GPU (if available) and set evaluation mode for inference without gradient computation.
# 
# **Inference Pipeline**
# 
# * Process test images through the fine-tuned model using the same preprocessing pipeline as training.
# 
# * Extract both predicted class indices and raw logits for comprehensive evaluation.
# 
# **Result Storage**
# 
# Collect predictions and logits in arrays for analysis. This enables comparison between original model performance and improvements from targeted data augmentation on challenging samples.

# In[ ]:


print("\nApplying retrained model to test set...")

# Load the best retrained model
final_model = ModernLeNet5()
final_model.load_state_dict(torch.load(retrain_model_save_path, map_location=device))
final_model = final_model.to(device)
final_model.eval()

# Apply retrained model to test set
retrained_predictions = []
retrained_logits = []

with torch.inference_mode():
    for images, _ in tqdm(test_loader, desc="Evaluating retrained model"):
        images = images.to(device)

        logits = final_model(images)
        retrained_logits.append(logits.cpu().numpy())

        _, predicted = torch.max(logits.data, 1)
        retrained_predictions.extend(predicted.cpu().numpy())

# Concatenate all results
retrained_logits = np.concatenate(retrained_logits, axis=0)

print(f"Retrained model evaluation complete.")
print(f"Predictions shape: {len(retrained_predictions)}")
print(f"Logits shape: {retrained_logits.shape}")



# ## Store Retrained Model Predictions in FiftyOne
# 
# Here's what we are doing next:
# 
# **Integrating Fine-tuned Model Results**
# 
# * Store retrained model predictions as [FiftyOne Classification](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Classification) objects in the test dataset.
# 
# * Each sample receives prediction label, confidence score, and raw logits for comprehensive analysis.
# 
# **Classification Object Creation**
# 
# Convert raw model outputs into structured FiftyOne format. Apply softmax to logits for confidence scores and map predicted indices to class labels using the established label mapping.
# 
# **Dataset Integration**
# 
# Add retrained predictions as a new field alongside existing CLIP and the original LeNet predictions. This enables direct comparison between all model variants within the same dataset framework.
# 
# **Analysis Preparation**
# 
# Structured storage enables filtering, querying, and evaluation using FiftyOne's built-in tools.
# 
# * [FiftyOne's Views Cheat Sheet](https://docs.voxel51.com/cheat_sheets/views_cheat_sheet.html)
# * [FiftyOne's Filtering Cheat Sheet](https://docs.voxel51.com/cheat_sheets/filtering_cheat_sheet.html)
# 
# With this we compare model performance and identify samples where fine-tuning improved or degraded predictions.

# In[ ]:


print("Storing fine-tuned model predictions in FiftyOne...")

for i, sample in enumerate(tqdm(test_dataset, desc="Storing retrained predictions")):
    predicted_idx = retrained_predictions[i]
    predicted_label = dataset_classes[predicted_idx]
    sample_logits = retrained_logits[i]

    confidences = Fun.softmax(torch.tensor(sample_logits), dim=0).numpy()
    predicted_confidence = float(confidences[predicted_idx])

    classification = fo.Classification(
        label=predicted_label,
        confidence=predicted_confidence,
        logits=sample_logits.tolist()
    )

    sample["retrained_lenet_classification"] = classification
    sample.save()

print("Retrained model predictions stored successfully!")



# ## Evaluate Retrained Model Performance
# 
# **Comprehensive Assessment of Fine-tuned Model**
# 
# Evaluate the retrained model using FiftyOne's classification evaluation framework. Generate detailed performance metrics including accuracy, precision, recall, and F1-scores for direct comparison with the original model.
# 
# 
# 
# 

# **Performance Comparison Analysis**
# 
# Compare original LeNet performance against the retrained model across key metrics. Calculate improvement deltas to quantify the impact of targeted data augmentation on model robustness.

# In[ ]:


print("\nEvaluating retrained model performance...")

# Evaluate retrained model
retrained_evaluation_results = test_dataset.evaluate_classifications(
    "retrained_lenet_classification",
    gt_field="ground_truth",
    eval_key="retrained_lenet_eval"
)

print("\n" + "="*60)
print("RETRAINED MODEL EVALUATION RESULTS")
print("="*60)
retrained_evaluation_results.print_report(digits=4)



# ## Compare Original vs Retrained Performance
# 

# In[ ]:


print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

# Get metrics for both models
original_metrics = lenet_evaluation_results.metrics()
retrained_metrics = retrained_evaluation_results.metrics()

print(f"{'Metric':<20} {'Original':<12} {'Retrained':<12} {'Improvement':<12}")
print("-" * 56)

metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']

for metric in metrics_to_compare:
    if metric in original_metrics and metric in retrained_metrics:
        orig_val = original_metrics[metric]
        retrain_val = retrained_metrics[metric]
        improvement = retrain_val - orig_val

        print(f"{metric:<20} {orig_val:<12.4f} {retrain_val:<12.4f} {improvement:+.4f}")



# ## Analysis of Misclassified Samples After Retraining
# 

# **Misclassification Analysis**
# 
# * Identify samples fixed by retraining versus those newly misclassified.
# 
# * Compute net improvement in correct predictions to assess overall effectiveness of the augmentation strategy.

# In[ ]:


# Find samples that were misclassified before but correct now
originally_wrong = test_dataset.match(
    F("lenet_classification.label") != F("ground_truth.label")
)

now_correct = originally_wrong.match(
    F("retrained_lenet_classification.label") == F("ground_truth.label")
)

print(f"\nSamples fixed by retraining: {len(now_correct)}")

# Find samples that were correct before but wrong now
originally_correct = test_dataset.match(
    F("lenet_classification.label") == F("ground_truth.label")
)

now_wrong = originally_correct.match(
    F("retrained_lenet_classification.label") != F("ground_truth.label")
)

print(f"Samples broken by retraining: {len(now_wrong)}")

# Net improvement
net_improvement = len(now_correct) - len(now_wrong)
print(f"Net improvement in correct predictions: {net_improvement}")



# ## Launch FiftyOne App for Insights on the  Results
# 
# 
# * Launch FiftyOne App with comparative views showing all model predictions. Enable exploration of specific cases where fine-tuning improved or degraded performance for model debugging.
# 
# * Create filtered view containing ground truth labels and predictions from both original and retrained models. This enables direct comparison between model variants within the same interface.
# 
# * Access interactive confusion matrices, per-class metrics, and sample-level analysis through FiftyOne's evaluation framework. Filter by prediction differences to identify which samples benefited from augmentation.
# 
# * Use FiftyOne's query language to create views of samples fixed by retraining, samples broken by retraining, or samples where both models agree or disagree with ground truth labels.

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/samples_fixed_by_fine_tuning.png?raw=true)

# In[ ]:


print("\nLaunching FiftyOne App to explore results...")

# Create a view showing the comparison
comparison_view = test_dataset.select_fields([
    "ground_truth",
    "lenet_classification",
    "retrained_lenet_classification"
])

session.view = comparison_view
session.refresh()
print(f"FiftyOne App URL: {session.url}")


# In[ ]:


# View the samples that are fixed by fine-tuning
session.view = now_correct
session.refresh()
print(session.url)


# In[ ]:


print("\n" + "="*60)
print("AUGMENTATION AND FINE-TUNING SUMMARY")
print("="*60)
print(f"Identified {len(mislabeled_train_images_view)} misclassified training samples")
print(f"Applied MNIST-appropriate augmentations (rotation, translation, elastic deform, etc.)")
print(f"Created {len(torch_augmented_dataset)} augmented training samples")
print(f"Retrained model for {retrain_epochs} epochs with combined dataset")
print(f"Evaluated performance on test set")
print(f"Net improvement: {net_improvement} correctly classified samples")


# ## Key Insights
# Careful data augmentation has the potential to help with misclassified samples: By specifically targeting the samples that our model struggled with during initial training, we can create additional training examples that help the model learn to handle these edge cases more effectively. However we must do this carefully in order to not damage the model.
# 
# Small rotations and elastic deformations are effective for handwritten digits: These transformations mimic natural variations in human handwriting without making digits ambiguous. The key is finding the right balance - enough variation to improve robustness, but not so much that a "6" looks like a "9".
# 
# Fine-tuning with augmented data can improve model robustness: Rather than training from scratch, fine-tuning the already-trained model with augmented versions of problematic samples allows us to specifically address weaknesses while preserving the knowledge already learned.
# 
# FiftyOne enables easy tracking of prediction changes across model versions: By storing predictions from both the original and retrained models in the same dataset, we can directly compare performance and identify which specific samples improved or degraded, enabling targeted analysis and further improvements.
# 
# 

# ## Takeaways
# 
# This tutorial demonstrates several fundamental principles:
# 
# - **Zero-shot vs. Supervised Learning**: Modern pre-trained models can often achieve competitive performance without task-specific training, but custom models allow for domain-specific optimization
# - **Embeddings as Universal Representations**: High-dimensional vectors capture semantic similarity and enable powerful analysis and visualization techniques  
# - **Visual Debugging**: FiftyOne's interactive capabilities make it easy to understand model behavior and identify improvement opportunities
# - **Data Quality Matters**: Systematic analysis of your dataset often leads to more significant performance gains than model architecture changes
# 
# 
# 
# 

# # Image Classification Getting Started Series Summary
# 
# This comprehensive series walks you through the core components of working with
# classification data in FiftyOne: from loading and visualizing datasets, to creating embeddings, evaluating models, and finding systematic errors.
# 
# ## Summary of Steps
# 
# ### Step 1: Understanding the MNIST Dataset
# 
# Explore the MNIST dataset structure and load it into FiftyOne. Learn about the 60,000 training images and 10,000 test images of handwritten digits, and understand why this dataset remains fundamental for classification research.
# 
# ### Step 2: Creating and Visualizing Image Embeddings
# 
# Learn how neural networks represent images as high-dimensional vectors. Generate embeddings with CLIP and visualize them using dimensionality reduction techniques like PCA and UMAP to understand image similarity and clustering.
# 
# ### Step 3: Zero-shot Classification with CLIP
# 
# Discover how modern vision-language models can classify images without explicit training on your dataset. Use pre-computed embeddings to perform efficient zero-shot classification with meaningful text prompts.
# 
# ### Step 4: Evaluating Dataset Quality through Embeddings
# 
# Assess your dataset's composition by analyzing embedding distributions. Learn to identify representative samples, outliers, and potential data quality issues using FiftyOne's analysis capabilities.
# 
# ### Step 5: Traditional Supervised Classification
# 
# Build a custom Convolutional Neural Network (LeNet-5) from scratch in PyTorch. Understand the fundamentals of supervised learning including convolutional layers, training loops, and optimization.
# 
# ### Step 6: Bridging FiftyOne and PyTorch
# 
# Master the integration between FiftyOne's dataset management and PyTorch's training capabilities. Convert datasets to DataLoaders while maintaining metadata and handling preprocessing efficiently.
# 
# ### Step 7: Model Comparison and Benchmarking
# 
# Compare your custom CNN against CLIP's zero-shot classification using FiftyOne's evaluation framework. Learn comprehensive evaluation metrics and statistical significance testing.
# 
# ### Step 8: Analyzing Model Predictions
# 
# Interpret model behavior by examining prediction confidence and identifying hard samples. Use logit analysis to find systematic misclassification patterns and debug model failures.
# 
# ### Step 9: Data Augmentation Strategies
# 
# Improve model performance through principled data augmentation. Learn which geometric transformations and elastic deformations help MNIST classification while avoiding destructive augmentations.
# 
# ### Step 10: Advanced Error Analysis
# 
# Create targeted views of model failures using FiftyOne's visualization capabilities. Identify false positives, false negatives, and the most problematic samples for systematic model improvement.
# 
# ---
# 
# This series is part of the **Getting Started with FiftyOne** initiative. For
# more tutorials, head to [FiftyOne Documentation](https://docs.voxel51.com/).
# 
# 

# ## Suggested Exercises
# 
# 1. **Sample Quality Analysis**: Notice that we computed the uniqueness, representativeness, and hardness of the samples on the training set. Can you create filtered views of other batches of retraining data based on them?
# 
# Try:
# * Selecting the most unique samples or those with highest hardness scores for additional augmentation
# * Filtering out the most ambiguous or questionable cases out of the fine-tuning training data
# 
# 2. **Active Learning**: Use the uniqueness, representativeness and hardness metrics to implement an [active learning pipeline](https://voxel51.com/blog/supercharge-your-annotation-workflow-with-active-learning) that selects the most informative samples for manual annotation or additional augmentation.
# 
# 3. **Dataset Exploration**: Apply these techniques to other classification datasets like [CIFAR-10](https://docs.voxel51.com/dataset_zoo/datasets.html#dataset-zoo-cifar10) or [Fashion-MNIST](https://docs.voxel51.com/dataset_zoo/datasets.html#dataset-zoo-fashion-mnist)
# 
# 4. **Architecture Comparison**: Implement and compare different CNN architectures (e.g. [Network in Network](https://arxiv.org/pdf/1312.4400), [ResNet](https://arxiv.org/pdf/1512.03385)) on MNIST
# 
# 5. **Transfer Learning**: Use pre-trained ImageNet models and fine-tune them for digit classification
# 
# 6. **Custom Augmentations**: Design and test novel augmentation strategies specific to handwritten digits, try augmenting the dataset [adding colors to the digits](https://paperswithcode.com/dataset/colored-mnist) and see how this change impacts the model's performance.
# 
# 7. **CLIP Model Variants and Prompting**: Experiment with different CLIP model variants available in FiftyOne's Model Zoo and compare their zero-shot performance on MNIST. Test how different text prompts affect accuracy across model sizes.

# 
# ## Resources and Further Reading
# 
# - [FiftyOne Documentation](https://docs.voxel51.com/)
# - [FiftyOne's Filtering Cheatsheet](https://docs.voxel51.com/cheat_sheets/filtering_cheat_sheet.html)
# - [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html)
# - [FiftyOne Dataset Zoo](https://docs.voxel51.com/user_guide/dataset_zoo/index.html)
# - [PyTorch Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
# - [CLIP Paper: Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)

# ## Next Steps
# 
# Now that you've completed the Image Classification Getting Started series, here are
# some suggested next steps to deepen your journey with FiftyOne:
# 
# -   **Explore Object Detection**  
#     Learn how to work with bounding boxes, evaluate detection models, and find annotation mistakes in object detection datasets.
# 
# -   **Try Multi-label Classification**  
#     Extend these concepts to scenarios where images can belong to multiple classes simultaneously, common in real-world applications.
# 
# -   **Experiment with FiftyOne Plugins**  
#     Enhance your workflow with powerful plugins for advanced augmentations, active learning tools, and integrations with annotation platforms.
# 
# -   **Connect with the Community**  
#     Share your findings, ask questions, or browse community projects on the
#     [FiftyOne Discord](https://community.voxel51.com) or
#     [GitHub Discussions](https://github.com/voxel51/fiftyone/discussions).
# 
# -   **Apply to Your Own Datasets**  
#     Adapt these workflows to your real-world classification projects. Whether
#     it's medical imaging, satellite analysis, or manufacturing quality control — FiftyOne supports diverse domains.
# 
# -   **Dive into Advanced Topics**  
#     Explore segmentation, video analysis, and 3D data in the
#     [official documentation](https://docs.voxel51.com/).
# 
# 

# 

# In[ ]:





# In[ ]:


# Create a view of samples with file paths in the bad_val_images_paths set
bad_val_images_view = val_dataset.match(
    fo.ViewField("filepath").is_in(bad_val_images_paths)
)

print(f"Created a view for bad validation images with {len(bad_val_images_view)} samples.")

# Set the session view to visualize these samples
session.view = bad_val_images_view
session.refresh()
print(f"\nFiftyOne App URL showing bad validation images: {session.url}")

