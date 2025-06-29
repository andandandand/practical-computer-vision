#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/andandandand/practical-computer-vision/blob/main/notebooks/Visualize_and_Cluster_Embeddings_with_FiftyOne.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Tutorial - Visualize Embedding Spaces with FiftyOne
# 
# #### Author: Antonio Rueda-Toicen
# **antonio.rueda.toicen 'at' hpi 'dot' de**
# 
# 
# [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
# 
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

# ## Overview

# In this notebook we learn how to visualize an embedding space using FiftyOne after applying [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) to reduce its dimensionality. We also use `sklearn` and `sklearn-extra` to produce clusters using the [K-means](https://en.wikipedia.org/wiki/K-means_clustering) algorithm.
# 
# ## Image data
# The folder with the image data and can be found [here](https://drive.google.com/drive/folders/1oZOMfxEYcrYctZSdx3NHO8KTc0ETrUFI?usp=drive_link).
# 
# You can add it to your own Google Drive by **right clicking on the folder name** -> **Organize** -> **Add Shortcut to Drive**. Select the **"All locations"** tab -> **My Drive** and then create a folder called `art_recommendation`. This will allow you to access the data without having to download it.  

# In[1]:


# Install fiftyone
get_ipython().system('pip install fiftyone==1.4.1 > /dev/null')


# In[ ]:


from google.colab import drive
drive.mount('/gdrive')


# In[ ]:


from pathlib import Path
import os
artist_name = 'Hokusai'
path = Path(f'/gdrive/MyDrive/art_recommendation/{artist_name}')


# In[ ]:


# here 'paintings' and 'paintings_embeddings.pickle' should both appear
os.listdir(path)


# In[ ]:


# Path where the images are
images_dir = path / "paintings"
images_dir


# In[ ]:


# Number of images
len(os.listdir(images_dir))


# ## Create a fiftyone dataset
# 
# If fiftyone, datasets are a collection of images to which we can add metadata (labels, quality metrics, embeddings, etc.).

# In[ ]:


import fiftyone as fo


dataset_name = dataset_name = f"{artist_name}_paintings"

# delete the dataset in case it exists already on the Colab instance
# (due to multiple evaluations of the code cell)
if fo.dataset_exists(dataset_name):
  fo.delete_dataset(dataset_name)


# In[ ]:


# this creates an empty dataset
dataset = fo.Dataset(dataset_name)
dataset


# In[ ]:


# We use the location of the images_dir to add samples to the dataset
dataset.add_dir(images_dir, dataset_type=fo.types.ImageDirectory)


# ## Connect embeddings to the dataset

# In[ ]:


import pickle
with open(path /'paintings_embeddings.pickle', 'rb') as f:
    embeddings = pickle.load(f)


assert len(embeddings) == len(os.listdir(images_dir))


# In[ ]:


# Every sample is an image, here we connect a sample to its embedding
for sample in dataset:
  sample["embedding"] = embeddings[Path(sample.filepath)]
  sample.save()

dataset.save()


# In[ ]:


dataset.view()


# ## Principal Component Analysis

# In[ ]:


import fiftyone.brain as fob

embeddings = dataset.values('embedding')

pca_brain_key = f"{dataset_name}_pca"

# Check if brain key exists
if dataset.has_brain_run(pca_brain_key):
    # Delete the brain key
    dataset.clear_brain_runs(pca_brain_key)
    dataset.save()

# Compute 2D representation
results = fob.compute_visualization(
    dataset,
    embeddings=embeddings,
    num_dims=2,
    method="pca",
    brain_key=pca_brain_key,
    verbose=True,
    seed=51,
)


# ## Clustering

# In[ ]:


import numpy as np
from sklearn.cluster import KMeans
import fiftyone as fo


# Choose the number of clusters
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Assign each sample its cluster label
for sample, label in zip(dataset, labels):
    sample["k_means_cluster"] = int(label)
    sample.save()

print("Clusters computed and stored in each sample's 'cluster' field.")


# In[ ]:


# Save the dataset to keep views in sync
dataset.save()


# ## Launch the FiftyOne App

# In[ ]:


# Launch the FiftyOne App, if you get an error try going into Runtime -> Run All
# If that fails, go to Runtime -> Disconnect and Delete Runtime
session = fo.launch_app(dataset, auto=False)


# In[ ]:


# Copy and paste the session url in a separate browser window
session.url

