#!/usr/bin/env python
# coding: utf-8

# # Tutorial - Visualizing Image Embeddings with TensorBoard
# #### Author: Antonio Rueda-Toicen
# **antonio.rueda.toicen 'at' hpi 'dot' de**
# 
# 
# [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
# 
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
# 
# This tutorial is part of a series on image similarity. In the previous notebooks we saw how to download images from Google Images, how to generate embeddings from them, and how to cluster them to select representative images and deduplicate a dataset.
# 
# In this notebook, we explore how to visualize the embeddings space using TensorBoard. This helps us better understand both our dataset and the method that was used to create the embeddings. Here we work with a dataset of artworks from Banksy that we have scraped from Google Images.
# 
# The folder with the downloaded images and pickle file with the embeddings is available [here](https://drive.google.com/drive/folders/1l7VvTnRwWpo4_wPPMrUcmMtKeaxuFTfY?usp=drive_link).
# Right click on the folder name and select "Add Shortcut to Drive" in order to add it to your own Drive.
# 

# In[ ]:


from google.colab import drive
drive.mount('/gdrive')


# In[ ]:


from pathlib import Path
path = Path('/gdrive/MyDrive/art_recommendation/banksy')


# In[ ]:


import pickle

with open(path / 'paintings_embeddings.pickle', 'rb') as f:
   paintings_embeddings = pickle.load(f)


# ### Using SummaryWriter to Visualize our Embeddings

# The `SummaryWriter` class in PyTorch's TensorBoard utility provides a high-level API for logging various types of data, making it possible to visualize these data using TensorBoard. TensorBoard itself is a web-based tool initially created for TensorFlow but has since been adapted for PyTorch as well. Here we use it only to visualize embeddings, although `SummaryWriter` has various applications when training neural networks. It's similar to the experiment tracking that we can find in tools [MLFlow](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html) or [Weights And Biases](https://docs.wandb.ai/guides/integrations/pytorch).
# 
# ### Key Features of `SummaryWriter`
# 
# 1. **Scalar Logging**: You can log single scalar metrics like loss and accuracy.
# 2. **Image Logging**: Visualize images, including the input data or feature maps.
# 3. **Histogram Logging**: Show the distribution of tensor values.
# 4. **Text Logging**: Log any textual information.
# 5. **Graph and Network Logging**: Visualize the model's computational graph.
# 6. **Embedding Logging**: Visualize high-dimensional data like word embeddings or image feature vectors. This is how we use it in this notebook.
# 
# ### Basic Usage of `SummaryWriter`
# 
# #### Initialization
# We first create a `SummaryWriter` object, optionally specifying the directory where we want the log files to be saved.
# 
# ```python
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/experiment1')
# ```
# 
# #### Writing Scalars
# We can log scalars like loss or accuracy at each step (if we were using `SummaryWriter` to monitor a training loop).  
# 
# ```python
# for epoch in range(100):
#     writer.add_scalar('Loss/train', loss_value, epoch)
#     writer.add_scalar('Accuracy/train', accuracy_value, epoch)
# ```
# 
# #### Writing Images
# Log images or feature maps.
# 
# ```python
# writer.add_image('Input Image', img_tensor, epoch)
# ```
# 
# #### Writing Embeddings
# You can log embeddings along with labels and images.
# 
# ```python
# writer.add_embedding(embedding_tensor, metadata=label_list, label_img=image_tensor, global_step=epoch)
# ```
# 
# #### Closing the Writer
# Once you're done with logging, you can close the writer.
# 
# ```python
# writer.close()
# ```
# 
# #### Launching TensorBoard
# After running the code, you can launch TensorBoard from a magic cell to visualize the logged data.
# 
# ```bash
# %load_ext tensorboard
# %tensorboard --logdir runs
# ```
# 
# This starts a web server on your machine, accessible by default at `http://localhost:6006/`. You can view various metrics and visualizations here.
# 
# `SummaryWriter` creates an event file in the specified directory (`runs/paintings_embeddings_experiment` in the example). TensorBoard reads this file to produce its visualizations. This makes it easy to track experiments and understand what's happening in your models.

# In[ ]:


# Import required libraries
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# Initialize an empty list to store the thumbnail images
thumbnail_images = []

# Define the size for the thumbnails
thumbnail_size = (32, 32)

# Initialize transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Loop over each image path to open, resize, and append to the list
for filename in paintings_embeddings.keys():
    with Image.open(filename) as img:
        img = img.resize(thumbnail_size)  # Explicitly resize the image
        thumbnail_images.append(img)

# Convert thumbnail images to PyTorch tensors
thumbnail_tensors = torch.stack([transform(img) for img in thumbnail_images])

# Convert embeddings to a PyTorch tensor
embeddings_tensor = torch.stack([torch.tensor(emb) for emb in paintings_embeddings.values()])

# Initialize a SummaryWriter
writer = SummaryWriter('runs/paintings_embeddings_experiment')

# Add embeddings and thumbnails to the writer
writer.add_embedding(
    embeddings_tensor,
    label_img=thumbnail_tensors,
    global_step=1,
    tag='paintings_embeddings'
)

# Close the writer
writer.close()


# ## Explore the dataset
# 
# Now we run Tensorboard in order to understand the dataset better. Notice how many near duplicates you find in the dataset and where are they spread how. How does the visualization change when we use [PCA](https://www.datacamp.com/tutorial/principal-component-analysis-in-python) vs [T-SNE](https://distill.pub/2016/misread-tsne/) or [UMAP](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)?

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs')


# ## References
# 
# * [Visualizing data using the embedding projector in TensorFlow](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin)
# 
# * [Tensorboard Projector](https://projector.tensorflow.org/)
