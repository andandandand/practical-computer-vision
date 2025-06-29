#!/usr/bin/env python
# coding: utf-8

# # Exploring Object Detection Performance with FiftyOne
# 
# In this notebook we compare the performance of a pretrained [RetinaNet](https://arxiv.org/abs/1708.02002) and a pretrained [Faster-RCNN model](https://arxiv.org/abs/1506.01497) on the [COCO 2017 validation set](https://cocodataset.org/#home), through the open source [FiftyOne](https://github.com/voxel51/fiftyone) SDK and visualization app.
# 
# It covers the following concepts:
# 
# - Loading a dataset with ground truth labels [into FiftyOne](https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/index.html)
# - [Adding model predictions](https://voxel51.com/docs/fiftyone/recipes/adding_detections.html) to your dataset
# - [Evaluating your model](https://voxel51.com/docs/fiftyone/user_guide/evaluation.html#detections) using FiftyOne's evaluation API
# - Viewing the best and worst performing samples in your dataset
# 
# ## Brief technical background
# 
# There is a big variety of object detection algorithms out there. Among those based on the filtering of region proposals (aka anchor boxes), there are two main families:
# 
# * Single shot detectors (SSDs) like RetinaNet and YOLO
# * Two stage detectors like Faster R-CNN
# 
# Two stage detectors filter out candidate region proposals before producing output detections. Single shot detectors evaluate all region proposals at once.
# SSDs have the reputation of being faster, while two stage detectors seem to work better on small objects. We will put this last statement to the test in this example.
# 
# **What's the takeaway?**
# 
# Aggregate measures of performance like [mean Average Precision](https://kili-technology.com/data-labeling/machine-learning/mean-average-precision-map-a-complete-guide) don't give us the full picture of your detection model. In practice, the limiting factor on your model's performance is often data quality issues that we need to **see** to address. FiftyOne is designed to make it easy to do just that.
# 
# ## Inspecting your datasets
# 
# Running the workflow presented here on your ML projects will help you to understand the current failure modes (edge cases) of your model and how to fix them, including:
# 
# - Identifying scenarios that require additional training samples in order to boost your model's performance
# - Deciding whether your ground truth annotations have errors/weaknesses that need to be corrected before any subsequent model training will be profitable
# 
# 
# This walkthrough demonstrates how to use FiftyOne to perform hands-on evaluation of your detection model.

# ## Install FiftyOne

# In[1]:


# Install the library, you will need to uncomment this on Colab
#!pip install fiftyone==1.5.2 > /dev/null


# In[2]:


#Check installed versions
import fiftyone as fo
print(f"FiftyOne version: {fo.__version__}")

import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F


# ## Dataset loading

# In[ ]:


dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    dataset_name="evaluate-detections-tutorial",
    # Take out the max_samples kwarg to get the full validation dataset
    max_samples=10
)
# Changes made to this dataset will not persist in memory
dataset.persistent = False



# ## Dataset cloning
# 
# Cloning a FiftyOne dataset before working with it is a best practice that creates a safe working copy while preserving your original data.
# 
# 

# In[4]:


# Creates a full copy of the dataset (in case we want to go back to the original)
working_dataset = dataset.clone()
# Changes done to the clone of the dataset will persist in memory and appear in multiple Python sessions
working_dataset.persistent = True
print("Dataset loaded.")


# In[5]:


# Compute metadata on every image on the dataset
# This includes height, width, size in bytes and type of the files
# overwrite=True forces these values to be recomputed for samples with existing fields
working_dataset.compute_metadata(overwrite=True)


# In[6]:


# Let's inspect the first sample of the dataset.
# This will show its filepath, tags, metadata, and any existing fields like 'ground_truth'.
# Let's check the metadata field that we just computed.
print(working_dataset.first().metadata)


# In[7]:


session = fo.launch_app(working_dataset, auto=False)
print(session.url)


# In[ ]:


## Create a view of the 10 random samples on our dataset
# We launch the FiftyOne app to start visualizing the dataset.
# fo.launch_app should be called only once per notebook.
session.view = working_dataset.limit(10)
print(session.url)


# ## Add predictions to our samples from two different object detection models
# 
# 

# In[10]:


retinanet_model = foz.load_zoo_model("retinanet-resnet50-fpn-coco-torch")
faster_rcnn_model = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")


# In[12]:


# necessary to use multiprocessing in MacOS with PyTorch
import multiprocessing as mp
mp.set_start_method('fork', force=True)

# Here we apply two different object detection models.
# RetinaNet is a single shot detector (like YOLO), known for its speed.
# Faster-RCNN is a two stage detector, known for its ability to detect objects
# in a big variety of sizes and aspect ratios with relatively low fine-tuning.
working_dataset.apply_model(retinanet_model, 
                            label_field="retinanet_predictions", 
                            num_workers=2)


# In[13]:


working_dataset.apply_model(faster_rcnn_model, 
                            label_field="faster_rcnn_predictions",
                            num_workers=2)


# In[14]:


# session.refresh() keeps the data state the FiftyOne Python SDK and the FiftyOne app consistent
session.refresh()
# We can now see the predictions on the dataset
print(session.url)


# In[11]:


# Resets the session; the entire dataset will now be shown
session.view = None
print(session.url)


# In[12]:


# Going back to showing the view
session.view = working_dataset.view()
session.refresh()
print(session.url)


# Now try inspecting the predictions with confidence > 0.75

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/predictions.png?raw=true)

# ## Confidence thresholding through aggregations

# In[14]:


retinanet_quantiles = working_dataset.quantiles(
    "retinanet_predictions.detections.confidence",
    [0.25, 0.5, 0.75, 0.9],
)
print(retinanet_quantiles)


# In[30]:


high_conf_view = working_dataset.filter_labels("retinanet_predictions",
                                                F("confidence") > retinanet_quantiles[-1],
                                                only_matches=False)


# Note the `only_matches=False` argument. When filtering labels, any samples that no longer contain labels would normally be removed from the view. However, this is not desired when performing evaluations since it can skew your results between views. We set `only_matches=False` so that all samples will be retained, even if some no longer contain labels.

# In[31]:


# Print some information about the view
print(high_conf_view)


# In[32]:


# Print a prediction from the view to verify that it has labels with confidence above the threshold
sample = high_conf_view.first()
sample.retinanet_predictions


# In[ ]:


# Load high confidence view in the App
session.view = high_conf_view
print(session.url)


# Try inspecting patches on the view
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/inspecting_patches.png?raw=true)

# ## Evaluate predictions
# 

# In[ ]:


# Evaluate the predictions in the `faster_rcnn` field of our `high_conf_view`
# with respect to the objects in the `ground_truth` field
results = high_conf_view.evaluate_detections(
    "retinanet_predictions",
    gt_field="ground_truth",
    eval_key="eval",
    compute_mAP=True,
)


# In[ ]:


# Get the 10 most common classes in the dataset
counts = working_dataset.count_values("ground_truth.detections.label")
classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

# Print a classification report for the top-10 classes
results.print_report(classes=classes_top10)


# In[ ]:


print(results.mAP())


# In[ ]:


# install ipywidgets if needed for the interactive plot below
get_ipython().system("pip install 'ipywidgets>=8,<9'")


# In[ ]:


plot = results.plot_pr_curves(classes=["person", "car"])
plot.show()


# ## Sample level analysis

# In[ ]:


# Our dataset's schema now contains `eval_*` fields from a confusion matrix
print(working_dataset)


# In[ ]:


# The dataset keeps track of the evaluations that we have run
print(working_dataset.list_evaluations())


# In[ ]:


print(working_dataset.get_evaluation_info("eval"))


# In[ ]:


# Load the view on which we ran the `eval` evaluation
eval_view = working_dataset.load_evaluation_view("eval")
print(eval_view)


# In[ ]:


# Our detections have helpful evaluation data on them
sample = high_conf_view.first()
sample.retinanet_predictions


# In[ ]:


# View the `iscrowd` attribute on a ground truth object
sample = working_dataset.first()
print(sample.ground_truth.detections[0])


# ## Evaluation patches

# So, now that we have a sense for the aggregate performance of our model, let's dive into sample-level analysis by creating an [evaluation view](https://voxel51.com/docs/fiftyone/user_guide/app.html#viewing-evaluation-patches).
# 
# Any evaluation that you stored on your dataset can be used to generate an [evaluation view](https://voxel51.com/docs/fiftyone/user_guide/app.html#viewing-evaluation-patches) that is a patches view creating a sample for every true positive, false positive, and false negative in your dataset.
# Through this view, you can quickly filter and sort evaluated detections by their type (TP/FP/FN), evaluated IoU, and if they are matched to a crowd object.
# 
# These evaluation views can be created through Python or directly in the App as shown below.

# In[ ]:


eval_patches = working_dataset.to_evaluation_patches("eval")
print(eval_patches)


# In[ ]:


# let's use this evaluation to find false positives with confidence above > .85
session_view = high_conf_view
print(session.url )


# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/false_positive.png?raw=true)

# ## View with best performing cases

# In[ ]:


# Show samples with most true positives
session.view = high_conf_view.sort_by("eval_tp", reverse=True)
print(session.url)


# ## View with the worst performing cases by false positives

# In[ ]:


# Show samples with most false positives
session.view = high_conf_view.sort_by("eval_fp", reverse=True)
print(session.url)


# ## View with the best performing cases by false negatives

# In[ ]:


# Show samples with most false negatives
session.view = high_conf_view.sort_by("eval_fn", reverse=True)
print(session.url)


# ## Filtering by bounding box area

# In[ ]:


# Compute metadata so we can reference image height/width in our view
dataset.compute_metadata()


# In[ ]:


#
# Create an expression that will match objects whose bounding boxes have
# area less than 32^2 pixels
#
# Bounding box format is [top-left-x, top-left-y, width, height]
# with relative coordinates in [0, 1], so we multiply by image
# dimensions to get pixel area
#
bbox_area = (
    F("$metadata.width") * F("bounding_box")[2] *
    F("$metadata.height") * F("bounding_box")[3]
)
small_boxes = bbox_area < 32 ** 2

# Create a view that contains only small (and high confidence) predictions
small_boxes_view = high_conf_view.filter_labels("retinanet_predictions", small_boxes)

session.view = small_boxes_view
print(session.url)


# In[ ]:


# Create a view that contains only small GT and predicted boxes
small_boxes_eval_view = (
    high_conf_view
    .filter_labels("ground_truth", small_boxes, only_matches=False)
    .filter_labels("retinanet_predictions", small_boxes, only_matches=False)
)

# Run evaluation
small_boxes_results = small_boxes_eval_view.evaluate_detections(
    "retinanet_predictions",
    gt_field="ground_truth",
)


# In[ ]:


# Get the 10 most common small object classes
small_counts = small_boxes_eval_view.count_values("ground_truth.detections.label")
classes_top10_small = sorted(small_counts, key=counts.get, reverse=True)[:10]

# Print a classification report for the top-10 small object classes
small_boxes_results.print_report(classes=classes_top10_small)


# ## Inspecting the crowd views

# In[ ]:


# View the `iscrowd` attribute on a ground truth object
sample = working_dataset.first()
print(sample.ground_truth.detections[0])


# In[ ]:


# Create a view that contains only samples for which at least one detection has
# its iscrowd attribute set to 1
crowded_images_view = high_conf_view.match(
    F("ground_truth.detections").filter(F("iscrowd") == 1).length() > 0
)

session.view = crowded_images_view
print(session.url)


# In[ ]:


# Evaluating the crowd by the number of false positives
session.view = crowded_images_view.sort_by("eval_fp", reverse=True)
print(session.url)


# ## Using the model to improve the dataset (active learning)

# In[ ]:


# Tag all highly confident false positives as "possibly-missing"
(
    high_conf_view
        .filter_labels("retinanet_predictions", F("eval") == "fp")
        .select_fields("retinanet_predictions")
        .tag_labels("possibly-missing")
)


# These tagged labels could then be sent off to our annotation provider of choice for review and addition to the ground truth labels. FiftyOne currently offers integrations for [Scale AI](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.scale.html), [Labelbox](https://voxel51.com/docs/fiftyone/api/fiftyone.utils.labelbox.html), and [CVAT](https://voxel51.com/docs/fiftyone/api/fiftyone.types.dataset_types.html?highlight=cvat#fiftyone.types.dataset_types.CVATImageDataset).
# 

# In[ ]:


# Export all labels with the `possibly-missing` tag in CVAT format
(
    dataset
        .select_labels(tags=["possibly-missing"])
        .export("./possibly_missing_labels", fo.types.CVATImageDataset)
)


# ## Summary
# 
# In this tutorial, we covered loading a dataset into FiftyOne and analyzing the performance of an out-of-the-box object detection model on the dataset.
# 
# **So, what's the takeaway?**
# 
# Aggregate evaluation results for an object detector are important, but they alone don't tell the whole story of a model's performance. It's critical to study the failure modes of your model so you can take the right actions to improve them.
# 
# In this tutorial, we covered two types of analysis:
# 
# - Analyzing the performance of your detector across different strata, like high confidence, small objects in crowded scenes
# - Inspecting the hardest samples in your dataset to diagnose the underlying issue, whether it be your detector or the ground truth annotations

# ## About this tutorial
# 
# This tutorial is based on [FiftyOne's documentation](https://docs.voxel51.com/tutorials/evaluate_detections.html). You will notice a couple of minor changes.
# 
# * Views for the app launch on their own window, this makes it easier for us to inspect the output of our views on the app.
# * We create a clone of the COCO dataset at the start of the notebook so that we can go back to its original state if we want
# * As an exercise I encourage to try our [integration with Ultralytics's YOLO] and compare it with RetinaNet and Faster RCNN(https://docs.voxel51.com/integrations/ultralytics.html).
# 
# 

# In[ ]:




