#!/usr/bin/env python
# coding: utf-8

# ## Creating and saving datasets from the FiftyOne app

# In[13]:


import fiftyone as fo
import fiftyone.zoo as foz
from pathlib import Path


# In[2]:


foz.list_downloaded_zoo_datasets()


# In[3]:


dataset = foz.load_zoo_dataset('coco-2017', split='validation')


# In[4]:


session = fo.launch_app(dataset, auto=False)
print(session.url)


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/save-current-filters-as-view.png?raw=true)

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/save-books-view.png?raw=true)

# In[ ]:


# Load the saved view
books_view = dataset.load_view("books-view")

# Now you can work with the view programmatically
print(f"View contains {len(books_view)} samples")


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/books-view-filter.png?raw=true)

# In[11]:


dataset.list_saved_views()


# In[12]:


# Load the saved view
books_view = dataset.load_saved_view("books-view")

# Now you can work with the view programmatically
print(f"View contains {len(books_view)} samples")


# In[ ]:


parent_path = Path("/Users/antonio/Documents/Projects/GettingStartedWithFiftyOne/local_run/")
dataset_dir = parent_path / "data/aerial_images_with_duplicates"


# In[18]:


# Save the view as a separate dataset
books_dataset = books_view.clone()

parent_path = Path("/Users/antonio/Documents/Projects/GettingStartedWithFiftyOne/local_run/")
export_dir = str(parent_path / 'data/coco-validation-books')

# Export the dataset to a directory
books_dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
    export_media=True,
    overwrite=True,
)

