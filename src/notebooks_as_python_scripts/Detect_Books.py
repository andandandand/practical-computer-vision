#!/usr/bin/env python
# coding: utf-8

# # Addressing labeling annotation errors in COCO-2017 through model output inspection and mistakenness

# # COCO-2017 Book Label Problems
# 
# The COCO-2017 validation set contains numerous incorrect "book" annotations. Objects like shelves, desks, album covers, and kitchen equipment are mislabeled as books.
# 
# ## Root causes
# 
# `Human Annotation Errors`: Crowdworkers made mistakes when annotating objects, often confusing rectangular items with actual books or labeling whole stacks or shelves of books as 'book'. 
# 
# `Vague Guidelines`: The original annotation instructions lacked precise guidelines about what constitutes a book versus other flat, rectangular objects.
# 
# `Quality Control Gaps`: Many errors passed through the review process, particularly for less distinctive object categories.
# 
# These mislabeled annotations create systematic bias in the dataset. Models trained on this data learn incorrect associations, leading to reduced performance on detection tasks.
# 
# ![](https://raw.githubusercontent.com/andandandand/practical-computer-vision/refs/heads/main/images/books_ground_truth.png)
# 
# 
# ## Using models to relabel 
# 
# We can use automated verification methods to identify and correct suspicious annotations based on object shape, context, and visual features. FiftyOne's [mistakenness](https://docs.voxel51.com/tutorials/detection_mistakes.html) metric provides a great starting point. 
# 
# While relabeling we can apply techniques that handle label noise better, such as confidence-based filtering or using the output of one or various models to produce soft labels. 
# 
# Clean datasets like [COCO-ReM](https://github.com/kdexd/coco-rem)have been created using ensembles of models and active learning to produce higher quality labeling. 
# 
# FiftyOne's [verified auto-labeling](https://arxiv.org/abs/2506.02359) addresses many of these issues in labeling for object detection, producing cleaned-up versions of the COCO-2017 dataset through model ensembling. 

# In[8]:


import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from pathlib import Path


# ## Import dataset

# In[3]:


parent_path = Path('/Users/antonio/Documents/Projects/GettingStartedWithFiftyOne/local_run/')
dataset_dir = str(parent_path / 'data/coco-validation-books')


# In[4]:


# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  
# Import the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    label_field="ground_truth",
)


# In[5]:


dataset.compute_metadata()


# In[15]:


session = fo.launch_app(dataset, auto=False)
print(session.url)


# In[7]:


faster_rcnn_model = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")

dataset.apply_model(faster_rcnn_model, 
                    label_field="faster_rcnn_predictions", 
                    batch_size=32)


# In[14]:


sample = dataset.first()
sample


# ## Label filtering and confidence thresholding through the FiftyOne app
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/full_threshold.png?raw=true)

# ### Tag labels to remove for a new ground truth
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/remove-labels.png?raw=true)
# 

# In[19]:


session.refresh()
print(session.url)


# ## Compute mistakenness

# `fob.compute_mistakenness()` creates the following attributes:
# 
# ## Ground Truth Field
# 
# * `mistakenness` (float): Likelihood that the object label is incorrect
# 
# * `mistakenness_loc` (float): Likelihood that the bounding box is inaccurate  
# 
# * `possible_spurious` (bool): True for unmatched ground truth objects likely to be spurious annotations
# 
# ## Predictions Field
# 
# * `possible_missing` (bool): True for confident predictions without matching ground truth, indicating likely missing annotations
# 
# # Sample-Level Fields
# 
# * `mistakenness`: Maximum mistakenness value among ground truth objects in the sample
# 
# * `possible_spurious`: Count of possible spurious ground truth objects in the sample
# 
# * `possible_missing`: Count of possible missing ground truth objects in the sample

# In[ ]:


# Compute mistakenness of annotations in `ground_truth` field using 
# predictions from `predictions` field as point of reference
fob.compute_mistakenness(dataset, "faster_rcnn_predictions", label_field="ground_truth")


# In[21]:


dataset.get_field_schema()


# In[18]:


session.refresh()
print(session.url)


# ### Follow-up
# 
# * [Start a CVAT session](https://docs.voxel51.com/tutorials/cvat_annotation.html) to correct the dataset's annotations. 

# In[ ]:




