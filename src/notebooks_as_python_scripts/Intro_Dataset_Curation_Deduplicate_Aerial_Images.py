#!/usr/bin/env python
# coding: utf-8

# # Intro to Dataset Curation with FiftyOne and CLIP (Part 1 of 2)

# In this notebook we explore the use of:
# 
# * FiftyOne's dataset curation SDK and visualization app 
# * Multimodal embeddings (text + image) from the [CLIP](https://arxiv.org/pdf/2103.00020) model 
# * Deduplication of images based on embeddings and cryptographic hashes
# 
# We do this by curating a dataset of aerial images from Google Earth View. Some of the images are duplicated through near and exact copies, and we will use FiftyOne to identify and remove them.
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/has_exact_duplicate.png?raw=true)
# 
# In part 2, we use the CLIP model to produce labels for the images, and then visualize the final result in FiftyOne.

# ## Imports

# In[1]:


import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import fiftyone.core.utils as fou
from pathlib import Path
import os
from PIL import Image
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import List, Tuple, Dict, Set
import pandas as pd


# ## Define paths for local folders
# 
# To reproduce, download the files from the [Google Drive folder](https://drive.google.com/drive/folders/1zkrteGx7HMWIpZWTuKArJKoocDa77dQH?usp=drive_link). 

# In[2]:


parent_path = Path("/Users/antonio/Documents/Projects/GettingStartedWithFiftyOne/local_run/")
dataset_dir = parent_path / "data/aerial_images_with_duplicates"
len(os.listdir(dataset_dir))


# https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#imagedirectory

# In[3]:


# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.ImageDirectory,
)


# ## Compute metadata

# In[4]:


dataset.compute_metadata()


# In[5]:


# We can add the num_pixels field to each sample
for sample in dataset: 
    sample["metadata.num_pixels"] = sample.metadata.width * sample.metadata.height
    sample.save()


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/metadata_aggregation.png?raw=true)

# In[8]:


session = fo.launch_app(dataset, auto=False, port=5252)
print(session.url)


# ## Produce similarity index based on CLIP embeddings 
# 
# Embeddings are computed and images have a defined similarity index to others based on them.

# In[ ]:


# Index images by similarity
image_index = fob.compute_similarity(
    dataset,
    model="clip-vit-base32-torch",
    brain_key="img_similarity",
    embeddings="clip_embeddings",
)


# In[10]:


# Refresh the FO app session to see the new brain key
session.refresh()


# In[8]:


# This index can be prompter for similar images through text queries
image_index.config.supports_prompts


# In[9]:


# Each sample now has a 'clip_embeddings' field
np.array(dataset.values('clip_embeddings')).shape


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/a_coast_search.png?raw=true)

# In[10]:


session.view = dataset.view()
print(session.url)


# ## Dimensionality reduction and uniqueness computation

# In[11]:


# Here we project the high-dimensional embeddings to a 2D space using PCA
vis_results = fob.compute_visualization(dataset,
                                        embeddings='clip_embeddings',
                                        method='pca',
               # We need the brain_key in order to access the run from the FiftyOne app
                                        brain_key='clip_pca')

# Here we compute how unique each sample is according to its embeddings (on its full dimensionality, not the 2D PCA projection)
fob.compute_uniqueness(dataset, embeddings="clip_embeddings", 
                       uniqueness_field="uniqueness")

session.refresh()


# ![]()

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/embeddings_for_aerial_images.png?raw=true)

# In[12]:


# Sort the dataset by uniqueness and display it in the FiftyOne app
session.view = dataset.view().sort_by("uniqueness", reverse=False)
print(session.url)


# ## Query images by their similarity neighborhood

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/similarity_neighorhood.png?raw=true)

# In[13]:


a_sample_id = dataset.first().id
images_similar_to_query = dataset.sort_by_similarity(a_sample_id, k = 5)
session.view = images_similar_to_query
print(session.url)


# ## Computing near-duplicates with FiftyOne's Brain
# 
# In most cases, this is all you need. 

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/near-duplicates-view.png?raw=true)

# In[37]:


## Computing near duplicates with FiftyOne's Brain
# Calculate near duplicates
# https://docs.voxel51.com/brain.html#near-duplicates
dup_index = fob.compute_near_duplicates(
    dataset,
    embeddings="clip_embeddings",
    # may need to change this distance measure for non-default mode: thresh=0.02,
    )

duplicates_view = dup_index.duplicates_view(
    type_field="dup_type",
    id_field="dup_id",
    dist_field="dup_dist",
)

session.view = duplicates_view
print(session.url)


# ## Custom logic using embeddings to deduplicate the dataset 
# 
# In the next sections, we define custom logic to deduplicate the dataset 

# In[14]:


embeddings = np.array(dataset.values("clip_embeddings"))
embeddings.shape


# In[15]:


# Here we compute the cosine similarity matrix for the embeddings
# This matrix will show how similar each image is to every other image
similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix.shape)
print(similarity_matrix)


# In[16]:


n = len(similarity_matrix)
# We subtract the identity matrix to avoid self-similarity
# (i.e., each image is perfectly similar to itself)
similarity_matrix = similarity_matrix - np.identity(n)
similarity_matrix


# In[17]:


np.mean(similarity_matrix), np.std(similarity_matrix)


# In[18]:


np.min(similarity_matrix), np.max(similarity_matrix)


# ### Add computation of max similarity in every sample

# In[19]:


# Create a copy of similarity matrix and set diagonal to -1 (or very low value)
similarity_no_self = similarity_matrix.copy()
np.fill_diagonal(similarity_no_self, -1)  # Mask self-similarities

# Get max similarity for each image
max_similarities = np.max(similarity_no_self, axis=1)

print(f"Max similarities shape: {max_similarities.shape}")
print(f"Max similarity values: {max_similarities[:10]}")  # First 10 values



# In[20]:


# Add max similarity as a field to each sample
id_map = [s.id for s in dataset.select_fields(["id"])]

# Get both max value and index of most similar image
similarity_no_self = similarity_matrix.copy()
np.fill_diagonal(similarity_no_self, -1)

max_similarities = np.max(similarity_no_self, axis=1)
most_similar_indices = np.argmax(similarity_no_self, axis=1)

for idx, sample in enumerate(dataset):
    sample["max_similarity"] = float(max_similarities[idx])
    sample["most_similar_image_id"] = id_map[most_similar_indices[idx]]
    sample.save()

print("Added max_similarity and most_similar_image_id fields to dataset")


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/Most_similar_images.png?raw=true)

# In[21]:


session.refresh()
session.view = dataset.view().sort_by("max_similarity", reverse=True)
print(session.url)


# ### Tagging samples for removal based on similarity threshold and image size 
# 
# Here we define custom logic to keep the image with bigger size out of the similarity neighborhood.
# 

# In[22]:


class ImageDeduplicator:
    """
    Expert image deduplication system using cosine similarity neighborhoods.

    The algorithm works by:
    1. Converting similarity matrix to adjacency graph
    2. Finding connected components (neighborhoods)  
    3. Selecting highest resolution image from each neighborhood
    """

    def __init__(self, similarity_matrix: np.ndarray, image_dimensions, 
                 similarity_threshold: float = 0.8):
        """
        Initialize the deduplicator.

        Args:
            similarity_matrix: Square cosine similarity matrix from sklearn
            image_dimensions: List of (height, width) tuples OR list of pixel counts
            similarity_threshold: Minimum similarity to consider images duplicates
        """
        self.similarity_matrix = similarity_matrix
        self.threshold = similarity_threshold
        self.n_images = len(image_dimensions)

        # Handle both tuple format (height, width) and integer format (pixel count)
        if isinstance(image_dimensions[0], (tuple, list)):
            # Format: [(height, width), ...]
            self.image_areas = np.array([h * w for h, w in image_dimensions])
        else:
            # Format: [pixel_count, ...] (already calculated areas)
            self.image_areas = np.array(image_dimensions)

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Ensure inputs are consistent and valid."""
        if self.similarity_matrix.shape[0] != self.similarity_matrix.shape[1]:
            raise ValueError("Similarity matrix must be square")

        if self.similarity_matrix.shape[0] != len(self.image_areas):
            raise ValueError("Number of images must match similarity matrix dimensions")

        if not 0 <= self.threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")

    def _create_adjacency_matrix(self) -> csr_matrix:
        """
        Convert similarity matrix to binary adjacency matrix.

        This is like creating a friendship network - if two images are similar
        enough (above threshold), they're "connected" in the graph.
        """
        # Create binary adjacency matrix (1 where similarity > threshold)
        adjacency = (self.similarity_matrix >= self.threshold).astype(int)

        # Remove self-connections (diagonal should be 0)
        np.fill_diagonal(adjacency, 0)

        # Convert to sparse matrix for efficiency with large datasets
        return csr_matrix(adjacency)

    def _find_duplicate_neighborhoods(self) -> Tuple[int, np.ndarray]:
        """
        Find connected components in the similarity graph.

        Each connected component represents a "neighborhood" of similar images.
        Uses Union-Find algorithm under the hood via scipy.

        Returns:
            n_components: Number of unique neighborhoods
            labels: Array where labels[i] is the neighborhood ID for image i
        """
        adjacency = self._create_adjacency_matrix()
        return connected_components(adjacency, directed=False)

    def _select_best_from_neighborhood(self, neighborhood_indices: List[int]) -> int:
        """
        Select the highest resolution image from a neighborhood.

        Args:
            neighborhood_indices: List of image indices in this neighborhood

        Returns:
            Index of the image with maximum area (height × width)
        """
        neighborhood_areas = self.image_areas[neighborhood_indices]

        # Find index with maximum area within this neighborhood
        best_local_idx = np.argmax(neighborhood_areas)
        return neighborhood_indices[best_local_idx]

    def deduplicate(self) -> Dict:
        """
        Perform complete deduplication process.

        Returns:
            Dictionary containing:
            - 'keep_indices': List of image indices to keep
            - 'remove_indices': List of image indices to remove  
            - 'neighborhoods': Details about each duplicate neighborhood
            - 'stats': Summary statistics
        """
        # Find duplicate neighborhoods
        n_components, labels = self._find_duplicate_neighborhoods()

        # Group images by neighborhood
        neighborhoods = {}
        for img_idx in range(self.n_images):
            neighborhood_id = labels[img_idx]
            if neighborhood_id not in neighborhoods:
                neighborhoods[neighborhood_id] = []
            neighborhoods[neighborhood_id].append(img_idx)

        # Process each neighborhood
        keep_indices = []
        remove_indices = []
        neighborhood_details = []

        for neighborhood_id, img_indices in neighborhoods.items():
            neighborhood_size = len(img_indices)

            if neighborhood_size == 1:
                # Single image - definitely keep
                keep_indices.extend(img_indices) 
                neighborhood_details.append({
                    'neighborhood_id': neighborhood_id,
                    'size': neighborhood_size,
                    'images': img_indices,
                    'kept': img_indices[0],
                    'removed': [],
                    'is_duplicate_group': False
                })
            else:
                # Multiple images - select best one
                best_idx = self._select_best_from_neighborhood(img_indices)
                removed_indices = [idx for idx in img_indices if idx != best_idx]

                keep_indices.append(best_idx)
                remove_indices.extend(removed_indices)

                neighborhood_details.append({
                    'neighborhood_id': neighborhood_id,
                    'size': neighborhood_size,
                    'images': img_indices,
                    'kept': best_idx,
                    'removed': removed_indices,
                    'is_duplicate_group': True,
                    'kept_area': self.image_areas[best_idx]
                })

        # Compile results
        results = {
            'keep_indices': sorted(keep_indices),
            'remove_indices': sorted(remove_indices),
            'neighborhoods': neighborhood_details,
            'stats': {
                'total_images': self.n_images,
                'images_to_keep': len(keep_indices),
                'images_to_remove': len(remove_indices),
                'duplicate_neighborhoods': sum(1 for n in neighborhood_details if n['is_duplicate_group']),
                'reduction_percentage': (len(remove_indices) / self.n_images) * 100
            }
        }

        return results

    def get_similarity_report(self, results: Dict) -> pd.DataFrame:
        """
        Generate a detailed report of duplicate neighborhoods.

        Args:
            results: Output from deduplicate() method

        Returns:
            DataFrame with neighborhood analysis
        """
        report_data = []

        for neighborhood in results['neighborhoods']:
            if neighborhood['is_duplicate_group']:
                # Calculate similarity statistics within this neighborhood
                img_indices = neighborhood['images']
                neighborhood_similarities = []

                for i, idx1 in enumerate(img_indices):
                    for idx2 in img_indices[i+1:]:
                        sim = self.similarity_matrix[idx1, idx2]
                        neighborhood_similarities.append(sim)

                report_data.append({
                    'neighborhood_id': neighborhood['neighborhood_id'],
                    'group_size': neighborhood['size'],
                    'kept_image': neighborhood['kept'],
                    'kept_area': neighborhood['kept_area'],
                    'removed_images': neighborhood['removed'],
                    'avg_similarity': np.mean(neighborhood_similarities),
                    'min_similarity': np.min(neighborhood_similarities),
                    'max_similarity': np.max(neighborhood_similarities)
                })

        return pd.DataFrame(report_data)


# In[23]:


# Image dimensions for each image
dimensions = dataset.values("metadata.num_pixels")

# Create deduplicator
deduplicator = ImageDeduplicator(
    similarity_matrix=similarity_matrix,
    image_dimensions=dimensions,
    similarity_threshold=0.89
)

# Run deduplication
results = deduplicator.deduplicate()
keep_these_images = results['keep_indices']
len(keep_these_images), len(results['remove_indices'])


# In[24]:


# Convert indices to sample IDs
id_map = list(dataset.values("id"))
keep_sample_ids = [id_map[idx] for idx in results['keep_indices']]
remove_sample_ids = [ id_map[idx] for idx in results['remove_indices'] ]

#
# Tag samples based on  deduplication results
for sample_id in keep_sample_ids:
    sample = dataset[sample_id]
    if "keep" not in sample.tags:
        sample.tags.append("keep")
        sample.save()

for sample_id in remove_sample_ids:
    sample = dataset[sample_id]
    if "remove" not in sample.tags:
        sample.tags.append("remove")
        sample.save()


session.view = dataset.view()
print(session.url)


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/keep_or_remove.png?raw=true)

# ## Dataset to export

# In[26]:


dataset_to_export = dataset.select(keep_sample_ids).clone()


# In[27]:


# Export the filtered dataset
export_dir = str(parent_path / "data/aerial_images_without_duplicates")
os.makedirs(export_dir, exist_ok=True)

dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.FiftyOneDataset,
    export_media=True,  # Include media files,
    overwrite=True  # Overwrite existing files if they exist
)
print(f"Dataset exported to: {export_dir}")


# ## Identifying exact duplicates with cryptographic hashes

# In[28]:


for sample in dataset:
    sample["file_hash"] = fou.compute_filehash(sample.filepath)
    sample.save()

print(dataset)


# In[29]:


# The cryptographic hash of the first sample's file
dataset.first().file_hash


# In[30]:


# Get all file hashes from the dataset
all_hashes = dataset.values("file_hash")
unique_hashes = set(all_hashes)

# Number of exact duplicates
len(all_hashes) - len(unique_hashes)


# In[31]:


hash_counts = Counter(all_hashes)
duplicated_hashes = {hash_val:count for hash_val, count in hash_counts.items() if count > 1}
print(f"Duplicated hashes: {duplicated_hashes}")


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/has_exact_duplicate.png?raw=true)

# In[32]:


# Tag all samples with duplicated file hashes
samples_with_duplicates = dataset.match(F("file_hash").is_in(duplicated_hashes))
samples_with_duplicates.tag_samples("has_exact_duplicate")


# In[33]:


session.view = dataset.view().sort_by("file_hash")
print(session.url)


# ## Using the deduplication plug-in

# In[34]:


get_ipython().system('fiftyone plugins download      https://github.com/voxel51/fiftyone-plugins      --plugin-names @voxel51/brain')


# In[35]:


get_ipython().system('fiftyone plugins download https://github.com/jacobmarks/image-deduplication-plugin')


# ## Exercise
# 
# * Compare the deduplication performance when you compute the embeddings using [Mobilenet](https://docs.voxel51.com/model_zoo/models.html#mobilenet-v2-imagenet-torch) or [Dino-V2](https://docs.voxel51.com/model_zoo/models.html#mobilenet-v2-imagenet-torch). How do the similarity thresholds for optimal deduplication change with these models? 

# ## Further reading 
# 
# * [FiftyOne examples - image deduplication](https://github.com/voxel51/fiftyone-examples/blob/master/examples/image_deduplication.ipynb)
# * [FiftyOne deduplication plugin](https://voxel51.com/blog/eliminate-image-duplicates-with-fiftyone)
# 
