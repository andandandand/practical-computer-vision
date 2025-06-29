#!/usr/bin/env python
# coding: utf-8

# # Image Neighborhoods and Clustering of Street Artwork
# #### Author: Antonio Rueda-Toicen
# **antonio.rueda.toicen 'at' hpi 'dot' de**
# 
# 
# This tutorial is part 3 of a series on image similarity. Refer to the [previous notebook](https://colab.research.google.com/drive/1yR4e3OVXe_QqCIYkmDhcT75300P9Db1H#scrollTo=c8520ef2) to get the image embeddings.
# 
# The folder with the downloaded images is available [here](https://drive.google.com/drive/folders/1l7VvTnRwWpo4_wPPMrUcmMtKeaxuFTfY?usp=drive_link). Right click on the folder name and select "Add Shortcut to Drive" in order to add to your own Drive.
# 

# ## Intro
# 
# Image embeddings, similarity metrics, and clustering on it are powerful techniques used in various domains, including **image analysis, recommendation systems, and data cleaning**. In this notebook, we'll explore how to use image similarity metrics to analyze and clean a dataset of artworks.
# 
# ### Image Similarity
# Image similarity measures how alike two images are and can be used to identify duplicates, variations, or related works within a dataset. It's an important step in preparing a clean and coherent dataset for further analysis or machine learning tasks. When building machine learning models, it's particularly important for us to eliminate duplicates between test and training sets, as this creates issues with [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)#:~:text=In%20statistics%20and%20machine%20learning,when%20run%20in%20a%20production), meaning that we can't trust the accuracy that the model reports.
# 
# ### Image Clustering
# When we ['cluster'](https://en.wikipedia.org/wiki/Cluster_analysis#:~:text=Cluster%20analysis%20or%20clustering%20is,in%20other%20groups%20(clusters).) data, we break into groups. We use a measure of similarity to do this division.
# 
#  When applying a clustering algorithm to image embeddings we group similar images together, creating groups of related content. It helps in understanding the underlying patterns, themes, or categories within a dataset. Clustering can reveal insights into the artistic styles, subjects, or other common features among the images.
# 
# ### K-Medoids Algorithm
# In this notebook, we'll use the [k-medoids algorithm](https://en.wikipedia.org/wiki/K-medoids), a robust clustering method that partitions the dataset into 'k' clusters. Unlike [k-means](https://en.wikipedia.org/wiki/K-means_clustering), k-medoids choose actual data points as cluster centers (medoids), making it less sensitive to outliers.
# 
# As the medoids are actual data points, they are the representatives of their clusters.
# 
# The animation below shows how the K-medoids algorithm converges to a solution.
# 

# In[ ]:





# ![](https://upload.wikimedia.org/wikipedia/commons/e/e1/K-Medoids_Clustering.gif)

# We'll apply k-medoids to explore the dataset and find the most representative works of the artist Bansky.
# 
# Let's dive into the process and begin with mounting Google Drive to access the necessary files.

# ## 1. Mounting Google Drive
# 
# In this section, we mount Google Drive to the Colab environment. This allows us to access files and data stored in Google Drive, which is essential if we are working with datasets or other resources stored there.

# In[ ]:


from google.colab import drive
drive.mount('/gdrive')


# ## 2. Importing Libraries and Setting Paths
# 
# We begin by importing the necessary libraries that enable us to handle images, perform clustering, and conduct other related tasks. Next, we define the paths to the folders containing the images and data we will work with. Organizing paths helps us manage files efficiently.

# In[ ]:


# We use pathlib to ease the manipulation of filenames and paths in Google Drive
import os
from pathlib import Path


# ## 3. Reading and Validating Files
# 
# In this part, we read the necessary files, including image embeddings, and validate them to ensure that they are in the correct format. Validation helps in detecting and handling potential issues early in the process.

# In[ ]:


path = Path('/gdrive/MyDrive/art_recommendation')
artist = 'banksy'
folder_path = path / artist
os.listdir(folder_path)


# In[ ]:


import pickle
with open(folder_path / 'paintings_embeddings.pickle', 'rb') as handle:
  paintings_embeddings = pickle.load(handle)


# In[ ]:


len(paintings_embeddings)


# In[ ]:


list_of_painting_names = os.listdir(folder_path / 'paintings')
# Check that we can read the proper paths to the images
full_paths_to_images = [folder_path / 'paintings' / painting for painting in list_of_painting_names ]
full_paths_to_images


# In[ ]:


# Check the individual file names
os.listdir(folder_path / 'paintings')


# In[ ]:


# Check that we can open a particular image from Google Drive
from PIL import Image
Image.open(full_paths_to_images[0])


# In[ ]:


# Check that the keys of the dictionary contain image paths
Image.open(list(paintings_embeddings.keys())[0])


# In[ ]:


# Check the number of paintings in our dataset
paintings_to_visualize = {painting:embedding for painting, embedding in paintings_embeddings.items() if painting in full_paths_to_images}
len(paintings_to_visualize)


# ## Visualize Pairs

# 
# Visualizing pairs of images can provide insights into the similarities and differences between them. This step often helps in understanding the characteristics of the images and how they relate to each other.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
  return dot(a, b)/(norm(a)*norm(b))

def print_cosine_similarity(embedding_a, embedding_b):
  cos = cosine_similarity(embedding_a, embedding_b)
  result = f'Cosine similarity = {cos:.2f}'
  print(result)

def show_pair(imag_a, imag_b):
  plt.subplot(121)
  plt.imshow(np.array(imag_a))
  plt.axis("off")

  plt.subplot(122)
  plt.imshow(np.array(imag_b))
  plt.axis("off")


# In[ ]:


len(paintings_to_visualize)


# In[ ]:


# We create double slider to visualize the similarity between every two images
# in the dataset
image_index_a = 221  #@param {type: "slider", min: 0, max: 317, step:1}

image_index_b = 61  #@param {type: "slider", min: 0, max: 317, step:1}

embed_a = list(paintings_to_visualize.values())[image_index_a]
embed_b = list(paintings_to_visualize.values())[image_index_b]
image_a = Image.open(list(paintings_to_visualize.keys())[image_index_a])
image_b = Image.open(list(paintings_to_visualize.keys())[image_index_b])

print_cosine_similarity(embed_a, embed_b)
show_pair(image_a, image_b)


# ## Find Clusters of the Most Similar Images
# 
# Exploring the dataset looking at the 9 nearest neighbors of each image will allow us to understand it better and to check for duplicates. Note that the closest 'neighbor' to every image is itself. It's cosine similarity should be 1.0 (or very-very-very close to 1.0).

# ## 5. Calculating and Visualizing Cosine Similarity
# 
# In this section, we calculate the cosine similarity between images, which helps us understand how similar two images are. We also include code to visualize the similarity between different pairs of images.

# In[ ]:


# Separate vectors and image filenames (for convenience)
X = []
images = []
for image, embedding in paintings_embeddings.items():
    X.append(embedding)
    images.append(image)


# In[ ]:


from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=9, metric='cosine').fit(X)
nbrs


# In[ ]:


# Get the distances between neighbohoods and the indices of similarity
neighbor_distances, neighbor_indices = nbrs.kneighbors(X)
neighbor_similarities = 1 - neighbor_distances
neighbor_similarities, neighbor_indices


# 
# Clustering is the process of grouping similar objects together. In this section, we apply clustering techniques to group images that are similar to each other. This can help in understanding the relationships between different images and identifying patterns within the dataset. In this case, we are interested in finding which are the most representative artworks in our dataset.

# In[ ]:


# We use this code to restructure the dataset and get indices of most similar images
neighborhoods_dict = {}
for i in range(len(neighbor_indices)):
    center_image = images[neighbor_indices[i, 0]]
    neighbors_list = []
    for j in range(0, len(neighbor_indices[0])):
        neighbor_image = images[neighbor_indices[i][j]]
        neighbor_similarity = round(neighbor_similarities[i][j],2)
        neighbors_list.append([neighbor_image, neighbor_similarity])
    neighborhoods_dict[center_image] = neighbors_list


# In[ ]:


paintings_with_full_path = list(neighborhoods_dict.keys())


# In[ ]:


import skimage.io as io
import matplotlib.pyplot as plt


def plot_neighborhood(neighbors_row):
  """
    This function allows us to visualize the 9 nearest neighbors of every image in
    the dataset.
    We see the cosine similarity between the query image and its neighbors
    in the title of each plot.
  """
  f, axarr = plt.subplots(3,3, figsize=(15,15))

  axarr[0][0].imshow(io.imread(folder_path /'paintings' / neighbors_row[0][0]))
  axarr[0][1].imshow(io.imread(folder_path /'paintings' / neighbors_row[1][0]))
  axarr[0][2].imshow(io.imread(folder_path /'paintings' / neighbors_row[2][0]))
  axarr[1][0].imshow(io.imread(folder_path /'paintings' / neighbors_row[3][0]))
  axarr[1][1].imshow(io.imread(folder_path /'paintings' / neighbors_row[4][0]))
  axarr[1][2].imshow(io.imread(folder_path /'paintings' / neighbors_row[5][0]))
  axarr[2][0].imshow(io.imread(folder_path /'paintings' / neighbors_row[6][0]))
  axarr[2][1].imshow(io.imread(folder_path /'paintings' / neighbors_row[7][0]))
  axarr[2][2].imshow(io.imread(folder_path /'paintings' / neighbors_row[8][0]))

  axarr[0, 0].set_title(neighbors_row[0][1])
  axarr[0, 1].set_title(neighbors_row[1][1])
  axarr[0, 2].set_title(neighbors_row[2][1])
  axarr[1, 0].set_title(neighbors_row[3][1])
  axarr[1, 1].set_title(neighbors_row[4][1])
  axarr[1, 2].set_title(neighbors_row[5][1])
  axarr[2, 0].set_title(neighbors_row[6][1])
  axarr[2, 1].set_title(neighbors_row[7][1])
  axarr[2, 2].set_title(neighbors_row[8][1])

  for i in range(3):
        for j in range(3):
            axarr[i, j].axis('off')

plot_neighborhood(neighborhoods_dict[paintings_with_full_path[0]])


# In[ ]:


# @title Visualizing Image Neighborhoods with a Slider Widget {run:'auto'}

slider_value = 223  #@param {type: "slider", min: 0, max: 317}
plot_neighborhood(neighborhoods_dict[paintings_with_full_path[slider_value]])


# ## De-duplicating the dataset
# 

# In[ ]:


import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Assuming `image_embeddings` is your dictionary: keys are image paths, and values are embeddings
image_embeddings = paintings_embeddings.copy()

# Convert embeddings to a Tensor for batch computation of cosine similarity
embeddings_tensor = np.stack(list(image_embeddings.values()))

# Compute pairwise cosine similarity
# For large numbers of embeddings, consider doing this in smaller batches to avoid memory issues
cos_sim_matrix = cosine_similarity(embeddings_tensor)

# Generate all pairs of image paths
image_paths = list(image_embeddings.keys())
pairs = [(image_paths[i], image_paths[j], cos_sim_matrix[i, j])
         for i in range(len(image_paths))
         for j in range(i+1, len(image_paths))]  # i+1 to avoid self-comparison

# Create a DataFrame
df = pd.DataFrame(pairs, columns=['image_a_path', 'image_b_path', 'cosine_similarity'])

# Show the DataFrame (or save to CSV, etc.)
df.head()


# In[ ]:


len(df)


# In[ ]:


len(paintings_embeddings)


# In[ ]:


import skimage.io as io

def show_pair(imag_a, imag_b):
  plt.subplot(121)
  plt.imshow(np.array(imag_a))
  plt.axis("off")

  plt.subplot(122)
  plt.imshow(np.array(imag_b))
  plt.axis("off")


print(df.iloc[0].cosine_similarity)
show_pair(io.imread(df.iloc[0].image_a_path), io.imread(df.iloc[0].image_b_path))


# In[ ]:


# Sort the DataFrame by 'cosine_similarity' in descending order
df_sorted = df.sort_values(by='cosine_similarity', ascending=False)




# In[ ]:


# @title {run:'auto'}


slider_value = 12  # @param {type: "slider", min: 0, max: 49454}

print(df_sorted.iloc[slider_value].cosine_similarity)
show_pair(io.imread(df_sorted.iloc[slider_value].image_a_path), io.imread(df_sorted.iloc[slider_value].image_b_path))


# ### Choosing a threshold for deduplication

# In[ ]:


duplicates_df = df[(df.cosine_similarity > 0.9)]
all_images = set(image_embeddings.keys())
duplicate_set = set(duplicates_df.image_b_path.values)
deduplicated_set = all_images - duplicate_set
len(deduplicated_set)


# In[ ]:


len(duplicate_set)


# In[ ]:


image_embeddings_dedup = {image_path : embedding for image_path, embedding in image_embeddings.items() if image_path not in duplicate_set}
len(image_embeddings_dedup)


# In[ ]:


# Separate vectors and image filenames (for convenience)
X = []
images = []
for image, embedding in image_embeddings_dedup.items():
    X.append(embedding)
    images.append(image)


# ## Clustering Images
# 
# Here we work with the K-medoids algorithm to select the most representative images in the dataset.

# In[ ]:


get_ipython().system('pip install scikit-learn-extra')


# In[ ]:


from sklearn_extra.cluster import KMedoids


kmedoids = KMedoids(n_clusters=10, random_state=0,
                    init='k-medoids++', metric='cosine')

kmedoids.fit(X)


# In[ ]:


# Here we get the medoid images (representatives) and the number of images
# that are most similar to them
image_clusters = dict(zip(images, kmedoids.labels_))
image_clusters


# ## 8. Additional Clustering and Visualization
# 
# This part of the notebook contains additional code related to clustering and visualization of images. It includes techniques such as k-medoids clustering and visualizing clusters of similar images.

# In[ ]:


images_array=np.array(images)
representative_embeddings = kmedoids.cluster_centers_
representative_images = images_array[kmedoids.medoid_indices_]


# In[ ]:


# Here we get the filenames of the most representative images
representative_images


# In[ ]:


## Visualize representative images (medoids)
f, axarr = plt.subplots(1, 10, figsize=(30,30))

for i in range(10):
  axarr[i].imshow(io.imread(folder_path /'paintings' /  representative_images[i]))
  axarr[i].axis('off')


# In[ ]:


len(representative_images)


# In[ ]:


# @title Slider Tool to Visualize Every Neighborhood of the Ten Representative Images {run:'auto'}
# @markdown * Do these images have duplicates?
# @markdown * Do you find them representative?
# @markdown * **What happens with a different threshold for deduplication?**

slider_value = 4  # @param {type: "slider", min: 0, max: 9}
plot_neighborhood(neighborhoods_dict[representative_images[slider_value]])



# ## Choosing an 'Optimal' Number of Clusters Based on the Elbow Method
# 
# 

# 
# Choosing the ""right"" number of clusters is useful for achieving meaningful clustering results. In this part, we analyze different clustering options using the elbow method. Selecting a small number of clusters helps to summarize the dataset and in forming interpretable groups of similar images.
# 
# We want to see find where the inflexion point happens.
# 
# * **Distortion:** It is calculated as the average of the squared distances from the cluster centers of the respective clusters.
# 
# * **Inertia:** It is the sum of squared distances of samples to their closest cluster center.
# 
# 
# Both of these methods usually give similar results. Both plots should show a decrease in intertia or distortion as we increase the number of clusters. We use the soft rule of choosing the number of clusters where we see "an elbow" (a big inflection) in both of these curves.
# 
# In this set of plots, we see that the biggest inflection point happens at around 20 clusters, so it might be "good enough" to use 20 images to describe the dataset. **Think about this not as a hard rule, but as a heuristic to simplify and reduce the time of your explorations.**
# 
# Read more about this [here](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/).

# In[ ]:


from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(5, 100, 2)

for k in K:
    kmedoidModel = KMedoids(n_clusters=k, random_state=0, init='k-medoids++',
                            metric='cosine')
    kmedoidModel.fit(X)

    distortions.append(sum(np.min(cdist(np.array(X), kmedoidModel.cluster_centers_,
                      'cosine'),axis=1)))
    inertias.append(kmedoidModel.inertia_)

    mapping1[k] = sum(np.min(cdist(np.array(X), kmedoidModel.cluster_centers_,
                 'cosine'),axis=1))
    mapping2[k] = kmedoidModel.inertia_


# In[ ]:


plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()


# In[ ]:


plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

