#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/andandandand/practical-computer-vision/blob/main/notebooks/Part_2_Zero_Shot_Classification_Aerial_Images_CLIP_Ensemble.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Intro to Dataset Curation with FiftyOne and CLIP (Part 2 of 2)
# 
# In this notebook we explore the use of:
# 
# * FiftyOne's dataset curation SDK and visualization app
# * Multimodal embeddings (text + image) from the [CLIP](https://arxiv.org/pdf/2103.00020) model to produce labels for images (aka Zero-shot classification)
# 
# We label dataset of aerial images from Google Earth View. This is the output of deduplication that we performed [in the previous notebook](https://github.com/andandandand/practical-computer-vision/blob/main/notebooks/Intro_Dataset_Curation_Deduplicate_Aerial_Images.ipynb).
# 
# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/clip_ensemble_labels.png?raw=true)
# 
# We will use an [ensemble of CLIP models](https://docs.voxel51.com/tutorials/zero_shot_classification.html) to produce majority vote labels for the images.
# 
# * OpenAI's base CLIP model
# * CLIPA
# * DFN
# * EVA-CLIP
# * MetaCLIP

# ## Install FiftyOne and openCLIP
# 

# In[ ]:


get_ipython().system('pip install fiftyone==1.5.2 > /dev/null')


# In[ ]:


get_ipython().system('pip install open_clip_torch==2.32.0 > /dev/null')


# ## Imports

# In[ ]:


import fiftyone as fo
import fiftyone.zoo as foz
import torch
from pathlib import Path
import os
import numpy as np


# ## Mount Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/gdrive')
get_ipython().run_line_magic('cd', '/gdrive')


# ## Define paths for local folders

# In[ ]:


parent_path = Path("/gdrive/MyDrive/fiftyone-getting-started-datasets/aerial-images")
os.listdir(parent_path)


# In[ ]:


aerial_images_path = parent_path / 'aerial_images_without_duplicates/data'
len(os.listdir(aerial_images_path))


# In[ ]:


# Check if dataset exists and delete it (dataset names are unique in FiftyOne)
dataset_name = "aerial-images-tagged"

if dataset_name in fo.list_datasets():
    print(f"Dataset '{dataset_name}' exists. Deleting...")
    fo.delete_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' deleted.")
else:
    print(f"Dataset '{dataset_name}' does not exist.")


# In[ ]:


# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=aerial_images_path,
    dataset_type=fo.types.ImageDirectory,
    name=dataset_name,
    persistent=True
)

dataset.compute_metadata(overwrite=True)


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[ ]:


text_prompt = 'an aerial photo of'

classes = ['a river',
           'a river in the jungle',
           'a river next to an urban area',
           'a river delta merging with the sea',
           'a body of water',
           'a body of water next to an urban area',
           'a jungle',
           'a forest',
           'a river in a forest',
           'a farmland',
           'a coast',
           'a desert',
           "a harbor next to a desert",
           'a desert next to a river',
           'a desert next to a body of water',
           'a desert next to an urban area',
           'a desert next to a coast',
           'a desert next to a forest',
           'terrain covered by snow',
           'a city',
           'an airport',
           'a sports stadium',
           'an urban area',
           'a city next to the coast',
           'military planes parked next to each other',
           'containers in a harbor',
           'ships in the ocean',
           'the ocean',
           'a beach',
           'a beach next to an urban area',
           'a mountainscape',
           'a refinery',
           'ships and containers in a harbor',
           'ships and boats in a harbor, next to an urban area',
           'dense vegetation next to a desert',
           'an island',
           'a harbor next to an urban area',
           'antartica or artic area, ice and water',
           'railroad tracks',
           'a train station',
           'a highway',
           'farming terraces',
           'an oil rig in the sea']


# In[ ]:


clip_model = foz.load_zoo_model(
    "clip-vit-base32-torch",
    text_prompt=text_prompt,
    classes=classes,
    device=device
)


# In[ ]:


print(f"The model is loaded on {clip_model._device}")


# In[ ]:


dataset.apply_model(
    model=clip_model,
    label_field="clip_zero_shot_classification",
    # This is how many samples we will show to the model at once
    batch_size=32,
    store_logits=True,
    progress_bar=True,
)


# In[ ]:


session = fo.launch_app(dataset, auto=False)
print(session.url)


# ## CLIP-variants

# In[ ]:


open_clip_args = {
    "clipa": {
        "clip_model": 'hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B',
        "pretrained": '',
        },
    "dfn": {
        "clip_model": 'ViT-B-16-quickgelu',
        "pretrained": 'dfn2b',
        },
    "eva02_clip": {
        "clip_model": 'EVA02-B-16',
        "pretrained": 'merged2b_s8b_b131k',
        },
    "metaclip": {
        "clip_model": 'ViT-B-32-quickgelu',
        "pretrained": 'metaclip_400m',
        },
    }


# In[ ]:


for name, args in open_clip_args.items():
    print(f"Applying {name} model")
    clip_model = args["clip_model"]
    pretrained = args["pretrained"]
    model = foz.load_zoo_model(
        "open-clip-torch",
        clip_model=clip_model,
        pretrained=pretrained,
        classes=classes,
        store_logits=True,
        batch_size=32,
        text_promopt=text_prompt
    )

    dataset.apply_model(model, label_field=name, save_logits=True)
    session.refresh()


# In[ ]:


session.view = dataset.view()
print(session.url)


# In[ ]:


dataset.get_field_schema()


# In[ ]:


sample = dataset.first()
sample


# In[ ]:


predictions_fields = ['clip_zero_shot_classification', 'clipa', 'dfn', 'eva02_clip', 'metaclip']
for sample in dataset:
    sample_labels = []
    confidences = []
    for prediction_field in predictions_fields:
       sample_labels.append(sample[prediction_field].label)
       confidences.append(sample[prediction_field].confidence)

    # Convert to numpy arrays
    labels_array = np.array(sample_labels)
    confidences_array = np.array(confidences)

    # Find unique labels and their counts
    unique_labels, counts = np.unique(labels_array, return_counts=True)

    # Find the maximum count and get all labels with that count
    max_count = np.max(counts)
    most_common_mask = counts == max_count
    most_common_labels = unique_labels[most_common_mask]

    most_common_label = most_common_labels[0]
    #print(f"Most common label: {most_common_label}")

    # Get indices for ONLY the first most common label
    indices = np.where(labels_array == most_common_label)[0]

    # Calculate mean confidence for this specific label only
    conf_mean = np.mean(confidences_array[indices])

    # Save the most common label and its mean confidence as a Classification
    sample['most_common_label'] = fo.Classification(label=most_common_label, confidence=conf_mean)
    sample.save()

session.refresh()


# ## Visualize the dataset with consensus labeling

# In[ ]:


# Launch the FiftyOne app to visualize the dataset
session.view = dataset.view()
print(session.url)


# In[ ]:


# Export to disk
export_dir = str(parent_path / "data/tagged_aerial_images")
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.FiftyOneDataset,
    export_media=True,  # Include media files,
    overwrite=True  # Overwrite existing files if they exist
)
print(f"Dataset exported to: {export_dir}")


# In[ ]:


# Import from disk
imported_dataset = fo.Dataset.from_dir(
    dataset_dir=export_dir,
    dataset_type=fo.types.FiftyOneDataset,
)


# In[ ]:


# Test that your custom field works correctly
print("Testing most_common_label field:")
for i, sample in enumerate(imported_dataset.take(3)):
    if hasattr(sample, 'most_common_label'):
        print(f"Sample {i+1}: {sample.most_common_label.label} (conf: {sample.most_common_label.confidence:.3f})")
    else:
        print(f"Sample {i+1}: No most_common_label field")


# ## Suggested Exercise
# 
# * Now that the images have been labeled and deduplicated, you can use the FiftyOne app to visualize the dataset. Try adding new features such as [clustering](https://docs.voxel51.com/tutorials/clustering.html) and [representativeness](https://docs.voxel51.com/brain.html#brain-image-representativeness) based on the embeddings.
