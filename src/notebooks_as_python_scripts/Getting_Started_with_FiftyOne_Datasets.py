#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/andandandand/practical-computer-vision/blob/main/notebooks/Getting_Started_with_FiftyOne_Datasets.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Getting started with FiftyOne
# 
# ## Content overview
# 
# In this tutorial we cover the following concepts:
# 
# - [Dataset basics and samples](https://beta-docs.voxel51.com/getting_started/basic/datasets_samples_fields/)
# - [Detection fields and labels](https://beta-docs.voxel51.com/api/fiftyone.core.labels.Detection.html)
# - [Dataset views and filtering](https://beta-docs.voxel51.com/how_do_i/cheat_sheets/filtering_cheat_sheet/)
# - [Computing and visualizing image embeddings with CLIP](https://docs.voxel51.com/tutorials/image_embeddings.html)
# 
# ## Who is this for
# 
# This tutorial is designed for:
# 
# * Computer vision practitioners who are new to the FiftyOne app
# * Anyone looking to integrate annotations into their visual datasets for analysis
# 
# ## Assumed knowledge
# 
# - Basic knowledge of image processing
# - Intermediate Python programming
# - Experience working with Jupyter notebooks
# 
# ## Estimated time to complete
# 30-45 minutes
# 
# ## Required packages for local installation
# 
# If not running from a Google Colab environment, we recommend using a virtual environment with [FiftyOne installed](https://beta-docs.voxel51.com/getting_started/basic/install/).
# 
# 

# Run this cell to install FiftyOne on Google Colab.

# In[1]:


get_ipython().system('pip install fiftyone==1.5.2 -q > /dev/null')


# Import the FiftyOne library and check the version that has been installed.

# In[2]:


import fiftyone as fo
fo.__version__


# ## Inspecting the quickstart `Dataset`
# 
# In the FiftyOne dataset zoo, we have a collection of publicly available datasets that can be easily loaded and used for computer vision tasks. The `quickstart` dataset is a small subset of 200 images from the COCO 2017 validation set with ground truth annotations and object detections that is great for demonstrating the features of FiftyOne.

# In[3]:


# Import the FiftyOne zoo for loading datasets.
import fiftyone.zoo as foz

#When we specify `persistent = True`, we make sure that changes to the dataset
# are saved across multiple Python sessions.
dataset = foz.load_zoo_dataset("quickstart", persistent=False)


# In case that we want to revert to the original state of the dataset,
# we can evaluate `dataset = dataset_clone` later in the notebook.

# In[4]:


# Creates a backup copy of the dataset (without copying the image files)
dataset_clone = dataset.clone()


# We can use the `stats()` method on a dataset to obtain info about its number of samples and size on disk.

# In[5]:


dataset.stats()


# The `info` field in the dataset can be used to add extra metadata. Here we are using it to specify where we got the dataset from and its license (Creative Commons 4.0).

# In[6]:


dataset.info["dataset_source"] = "https://docs.voxel51.com/dataset_zoo/datasets.html#dataset-zoo-quickstart"
dataset.info["dataset_license"] = "CC-BY-4.0"
dataset.info


# `.count()` tells us how many samples have been added to the dataset.

# In[7]:


dataset.count()


# In FiftyOne, a sample is an image and all its associated tags, metadata, and annotations.

# In[8]:


sample=dataset.first()
sample


# ## FiftyOne sample fields
# 
# In FiftyOne, a field is an attribute associated with each sample (e.g., image or video) in a dataset. Fields store labels, metadata, predictions, or custom data. Fields provide a way to organize and access information about your data within the FiftyOne framework. You can use fields to filter, sort, and analyze your dataset. Fields play a role in tasks like model evaluation and data visualization. Some examples of built-in fields are `filepath`, `ground_truth`, and `predictions`, but you can also define your own custom fields to store any data you need.
# 
# 

# Here we can inspect the fields of the sample.

# In[9]:


sample.field_names


# The ID of the sample is a hash that is unique for each image.

# In[10]:


sample.id


# We have a datetime object specifying when the dataset was created by us on disk.

# In[11]:


sample.created_at


# `sample.ground_truth` specifies the labels and positions of our object detections.

# In[12]:


sample.ground_truth


# `sample.predictions` will give us the list of all bounding boxes that have been computed on the dataset already. Note that each detection has a unique hash ID, a label, a confidence level, and an associated bounding box. Note that we are accessing the first sample from the *clone*. Later in the notebook we will delete the predictions field from the original sample. This cell is accessing the backup version of the dataset.

# In[13]:


sample = dataset_clone.first()
dataset = dataset_clone 
dataset.first()


# Notice that the `sample.filepath` points to the path to the image on the hard drive. `filepath` is the only required sample field when creating our own datasets.

# In[14]:


sample.filepath


# As the file is local, we can open it with PIL, NumPy or PyTorch.

# In[15]:


from PIL import Image
original_sample = Image.open(sample.filepath)
original_sample


# We can add fields to our sample, in order to extend it.

# In[16]:


sample['inspected_in_notebook'] = True
sample.field_names


# We can also add the field at the dataset level.

# In[17]:


dataset.add_sample_field("inspected_in_notebook", fo.BooleanField)


# In FiftyOne, files are loaded from disk. The images from the `quickstart` dataset are now on the hard drive of the computer running this notebook.

# In[18]:


from pathlib import Path
get_ipython().system('ls {Path(sample.filepath).parent}')


# When evaluating the dataset object, we get a quick look at its attributes.

# In[19]:


dataset


# Our dataset consists of images and we have 200 samples of them.

# In[20]:


dataset.media_type, len(dataset)


# ## Adding samples to the dataset

# Let's create a grayscale variation of the image above

# In[21]:


grayscale_sample = Image.open(sample.filepath).convert("L")
grayscale_sample


# We save it on the same location as the original file.

# In[22]:


save_folder = Path(sample.filepath).parent
grayscale_sample_filename = save_folder / "grayscale.png"
grayscale_sample.save(grayscale_sample_filename)


# In[23]:


# Create a new sample for the grayscale image
grayscale_sample_entry = fo.Sample(filepath=str(grayscale_sample_filename))

# Add the new sample to the dataset
dataset.add_samples([grayscale_sample_entry])

# Verify that the sample was added
print(f"Number of samples after adding grayscale: {len(dataset)}")


# The new sample automatically receives an id from a hash function.

# In[24]:


grayscale_sample_entry.id


# Let's mark it as inspected inside the notebook.

# In[25]:


grayscale_sample_entry.inspected_in_notebook = True


# Launch the FiftyOne App and load the full dataset into it.

# In[26]:


session = fo.launch_app(dataset, auto=False)
print(f"Click on the URL to access the FiftyOne app {session.url}")


# ## Removing fields from a dataset
# 
# We have used the FiftyOne app to inspect the ground truth and the pre-computed predictions from an object detection model.
# 
# To remove the ground_truth and predictions fields from your dataset, we can use the `delete_sample_fields()` method. This method takes a list of field names as input and removes them from all samples in the dataset.

# In[27]:


# Inspect the field schema of the dataset
dataset.get_field_schema()


# In[28]:


dataset.get_field_schema().get("predictions") is None


# In[29]:


# This operation happens in-place, it can be evaluated only once
if dataset.get_field_schema().get("predictions") is not None:
    dataset.delete_sample_field("predictions")
if dataset.get_field_schema().get("uniqueness") is not None:
    dataset.delete_sample_fields("uniqueness")


# In[30]:


dataset


# In[31]:


# The fields are no longer visible through the session
session = fo.launch_app(dataset, auto=False)
print(session.url)


# # Clone vs. Views in FiftyOne Datasets
# 
# In FiftyOne, both clones and views provide ways to work with datasets, but they serve different purposes:
# 
# ## Clones
# 
# A clone is a complete copy of a dataset. When we clone a dataset in FiftyOne:
# 
# - Changes made to the clone do not affect the original dataset
# - Clones are independent datasets with their own names in FiftyOne's MongoDB database
# - All samples, fields, and metadata are fully copied into a clone
# 
# Here's how we would create a clone:
# 
# 
# 

# Create a named clone.

# In[32]:


cloned_dataset = dataset.clone("my-quickstart-clone")


# The named clone will now appear in the list of available datasets, alongside the unnamed clone that we created at the start of the notebook and the quickstart.

# In[33]:


fo.list_datasets()


# In[34]:


cloned_dataset.first().filepath


# ## Views
# 
# A view is a filtered subset of the FiftyOne dataset. When when we create a view:
# 
# - Changes to samples in the view will affect the original dataset
# - It's not a separate dataset but a lens into the original dataset
# - No data is duplicated; it's just a different way to access the source dataset
# - Views are memory-efficient since they don't copy data
# - Views can apply filters, sorting, and other operations to show only specific data
# 
# Views are useful to retrieve a subset of data, it's good to check out the [views cheat sheet](https://docs.voxel51.com/cheat_sheets/views_cheat_sheet.html) to get a feel for what we can do with them.
# 
# 
# 

# ## Slicing to create views

# Slicing is a way to create dataset views. Here we select three samples from the dataset, starting at index 7.

# In[35]:


dataset[7:10]


# # `ViewField` in FiftyOne
# 
# A `ViewField` in FiftyOne is a dynamic field that computes its values on-the-fly rather than storing them directly in the dataset. Unlike regular fields that permanently store data, ViewFields are computed when accessed and don't persist any values to disk.
# 
# ## Key characteristics of ViewFields
# 
# 1. **Dynamic Computation**: Values are generated at access time through a user-defined function
# 2. **Non-persistent**: The values aren't stored in the database
# 3. **Memory Efficient**: Since values aren't stored, they don't increase dataset storage requirements
# 4. **Function-based**: Each `ViewField` is backed by a Python function that determines its values
# 
# ## How ViewFields work
# 
# When you define a `ViewField`, you provide a function that specifies how to compute the field's value for each sample. The function typically takes a sample as input and returns the desired value based on other fields in that sample.
# 
# Here's a basic example of creating a `ViewField` to filter out the images with cats in our quickstart data.
# 

# ### Using `match_labels()` to filter samples
# 
# The `match_labels()` method is used to create a `DatasetView` containing only the **samples** that have *at least one* label matching a specified filter condition within designated label fields. In the cell below you see it being used to select images labeled as "cat".

# In[36]:


from fiftyone import ViewField as F

# Use match_labels to filter samples that contain at least one "cat" label
cats_view = (
    dataset
    .match_labels(fields="ground_truth", filter=F("label") == "cat") # Keep only samples with "cat" detections
)

# Optional: You can verify the number of samples in the view
print(f"Number of samples containing cats: {len(cats_view)}")


# We can always re-launch the app with a view, allowing us to see the filtered dataset.

# In[37]:


session = fo.launch_app(cats_view, auto=False)
print(session.url)


# ## Differences between a dataset clone and a view
# 
# Think of a dataset as a library of books:
# - **Clone**: Making a complete duplicate of the library in a new building. Changes to one library don't affect the other.
# 
# - **View**: Creating a reading list that references specific books in the original library. If you write notes in a book from the reading list, those notes appear in the original library's book too.
# 
# ## When to use each
# 
# - Use **clones** when you need a complete backup or want to make extensive changes without affecting the original data. At the start of the of this notebook we created a clone of the original dataset in case that we want to revert back to it. Remember that the clone **doesn't** copy the media files (images, videos), only the fields from the FiftyOne dataset, so they are relatively cheap in terms of memory usage.
# - Use **views** when you need to temporarily filter, sort, or manipulate a dataset without duplicating data or when you want changes to propagate to the original dataset
# 

# ## Stats on a `DatasetView`

# We can use `stats()` on a dataset view to retrieve info on a subset of the data.

# In[38]:


cats_view.stats()


# # Computing and visualizing embeddings with CLIP
# 
# Embeddings are powerful numerical representations of unstructured data like images. Models like CLIP (Contrastive Language-Image Pre-Training) can generate embeddings that capture semantic meaning. By visualizing these embeddings in a lower-dimensional space (e.g., using PCA, UMAP or t-SNE), we can explore similarities and patterns in our dataset.
# 
# FiftyOne's Brain provides tools to compute and visualize embeddings. We'll use the `compute_visualization()` method with a CLIP model.

# Now, import the `fiftyone.brain` module.

# In[39]:


import fiftyone.brain as fob


# We will use `fob.compute_embeddings()` to generate both embeddings for our `dataset` using the `clip-vit-base-patch32` model. We use `fob.compute` and a two-dimensional projection of these embeddings that we can use to inspect the dataset.
# 
# This function will:
# 
# 1. Download the CLIP model from the FiftyOne Model Zoo (if not already downloaded).
# 2. Compute an embedding vector for each sample in the dataset.
# 3. Apply a dimensionality reduction technique to project the high-dimensional embeddings into a 2D space.
# 4. Store these 2D coordinates in a new field on each sample.
# 
# The `brain_key` parameter assigns a name to this computation, allowing us to reference it later.

# In[40]:


# Load a model with embeddings
model = foz.load_zoo_model("clip-vit-base32-torch")

# Generate embeddings for each sample
embeddings = dataset.compute_embeddings(model)


# In[41]:


# Compute visualization in two dimensions
two_dim_projection = fob.compute_visualization(
    dataset, embeddings=embeddings, seed=51, brain_key="embeddings_2d", method="pca"
)


# In[42]:


# Compute uniqueness of embeddings
fob.compute_uniqueness(dataset, embeddings=embeddings)


# In[43]:


# Launch the session
session = fo.launch_app(dataset, auto=False)
print(f"The app is available at {session.url}")


# ### Visualizing embeddings in the FiftyOne app
# 
# With the visualization data computed, we can now launch the FiftyOne App to explore the embeddings. The App provides an interactive scatter plot where each point represents an image.
# 
# After running the cell below, the FiftyOne App will launch or update.
# 1. Look for an **Embeddings** panel or a **brain icon** (usually on the left sidebar or as a tab).
# 2. In this panel, you should find your `brain_key` (e.g., `clip_quickstart_viz`) listed under available plots.
# 3. Click on it to load the 2D scatter plot of your image embeddings.
# 
# You can then explore the image neighborhood according to attributes. The [`uniqueness`](https://docs.voxel51.com/tutorials/uniqueness.html)attribute allows interesting explorations of our samples.
# 
# ![](https://raw.githubusercontent.com/andandandand/practical-computer-vision/refs/heads/main/images/uniqueness_3.png)
# 
# To access this view click on the `+` symbol next to samples, then `Curate -> Embeddings` and Select the brain key that we used to compute the projection. Then `"Split Horizontally"` to access the view of the embeddings next to the images. You can then use the lasso selection tool to highlight portions of the dataset.

# In[44]:


# Sort in increasing order of uniqueness (least unique first)
dups_view = dataset.sort_by("uniqueness")

# Open view in the App
session.view = dups_view
print(f"Duplicate views at: {session.url}")


# ## Summary
# 
# This tutorial covered:
# 
# - Loading datasets from the [FiftyOne Dataset Zoo](https://docs.voxel51.com/dataset_zoo/index.html).
# - Inspecting dataset samples and fields.
# - Adding samples to a dataset.
# - Cloning datasets for independent copies.
# - Creating dataset views for filtering and sorting.
# - Using [`ViewField`]() for dynamic computations.
# - Using [match_labels()](https://docs.voxel51.com/api/fiftyone.core.collections.html#fiftyone.core.collections.SampleCollection.match_labels) to filter samples.
# - Computing and visualizing image embeddings using [CLIP](https://docs.voxel51.com/dataset_zoo/index.html) and [compute_visualization](https://docs.voxel51.com/api/fiftyone.brain.visualization.html#fiftyone.brain.visualization.compute_visualization) from `fiftyone.brain`.
# - Launching the FiftyOne app from different dataset subsets and views.
# 
# 
