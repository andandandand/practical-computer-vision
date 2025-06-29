#!/usr/bin/env python
# coding: utf-8

# # Embeddings-based dataset curation for jaguar re-identification with FiftyOne
# 
# 
# This notebook utilizes the [FiftyOne](https://voxel51.com/docs/fiftyone/) library for various computer vision dataset analysis and manipulation tasks. The key operations performed with FiftyOne include:
# 
# *   **Dataset Creation/Loading**: Converting a Hugging Face dataset into a FiftyOne dataset format for enhanced analysis and visualization.
# *   **Embedding Computation**: Generating image embeddings using pre-trained models from the FiftyOne Model Zoo (e.g., DINOv2, MobileNet). These embeddings serve as rich feature representations of the images and allows to characterize the images by uniqueness and representativeness and deduplicate the dataset before preparing train / validation / and testing splits for a classification model. 
# *   **Data Visualization**:
#     *   Creating 2D visualizations (e.g., PCA, UMAP plots) from the computed embeddings to explore dataset structure, identify clusters, and find outliers.
#     *   Generating bar plots to understand the distribution of labels (e.g., individual jaguar IDs) within the dataset.
# * **Creation of Segmentation Masks with YOLOE**:
#     * We use FiftyOne's integration with ultralytics's YOLOE to compute segmentation masks for the heads and bodies of the jaguars. We inspect the quality of the segmentation masks through the FiftyOne app.      
# *   **Interactive Exploration**: Leveraging the FiftyOne App to visually inspect samples, their metadata, associated embeddings, and plots.
# *   **(Optional) Exporting/Sharing**: Preparing and potentially pushing the curated FiftyOne dataset or its derivatives to platforms like the Hugging Face Hub.
# 
# The overall goal is to use FiftyOne's powerful toolkit to gain insights into the jaguar image dataset and prepare it for model training.

# In[6]:


import os
from dotenv import load_dotenv
import fiftyone as fo
from PIL import Image
from datasets import load_dataset
from ultralytics import YOLOE
import fiftyone.zoo as foz



# Load environment variables from .env file
load_dotenv('../.env')

# Access the Hugging Face token
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("HF_TOKEN loaded successfully.")
    # You can now use hf_token in your code
else:
    print("HF_TOKEN not found in .env file or environment variables.")


# ## Load the dataset from HuggingFace

# In[ ]:


new_cache_dir = '../datasets'

hf_dataset = load_dataset("jaguaridentification/jaguars", 
                       cache_dir=new_cache_dir, 
                       split='cropped_body') 
# possible values: 'cropped_body', 'cropped_head', 'raw_images', 'segmented_body'


# In[7]:


hf_dataset.size_in_bytes / (1024**3)  # Convert bytes to GB


# In[8]:


fiftyone_image_dir = os.path.join(new_cache_dir, "fiftyone_images_jaguars")
os.makedirs(fiftyone_image_dir, exist_ok=True)

# 2. Create a new FiftyOne dataset
#    You can choose any name for your FiftyOne dataset.
#    If a dataset with this name already exists and is persistent, it will be loaded.
#    Otherwise, a new one is created.
try:
    fo_dataset = fo.load_dataset("jaguars-from-hf")
    fo_dataset.delete() # Delete if exists to start fresh for this example
    print("Deleted existing 'jaguars-from-hf' dataset.")
except ValueError:
    print("No existing 'jaguars-from-hf' dataset found. Creating a new one.")
    pass # Dataset doesn't exist, which is fine

fo_dataset = fo.Dataset("jaguars-from-hf")
fo_dataset.persistent = True # Make it persistent so it's saved in the database

# 3. Iterate through the Hugging Face dataset and add samples to FiftyOne
samples = []
for i, item in enumerate(hf_dataset):
    pil_image = item['image']  # Assuming 'image' field contains PIL.Image object
    label = item['label']      # Assuming 'label' field contains the class label

    # Define a filepath for the image
    # Ensure unique filenames, e.g., by using an index or an ID from the dataset if available
    image_filename = f"jaguar_{i:06d}.jpg" # Using an index for unique names
    filepath = os.path.join(fiftyone_image_dir, image_filename)

    # Save the PIL image to the filepath
    # Make sure the image format is correct (e.g., JPEG, PNG)
    try:
        pil_image.save(filepath)
    except Exception as e:
        print(f"Warning: Could not save image {filepath}. Error: {e}")
        # If image is already a path or has a 'filename' attribute, you might use that directly
        # For example, if item['image'].filename exists and is a valid path:
        # filepath = item['image'].filename
        # However, Hugging Face Image fields usually load PIL objects.
        # If saving fails, this sample will be skipped or have a missing image.
        continue


    # Create a FiftyOne sample
    sample = fo.Sample(filepath=filepath)

    # Add classification label (assuming it's a classification task)
    # Convert label to string if it's not already (e.g., if it's an int ClassLabel)
    sample["ground_truth"] = fo.Classification(label=str(label))

    # You can add other metadata from your Hugging Face dataset as well
    # For example, if you have bounding boxes:
    # sample["detections"] = fo.Detections(detections=[...])

    samples.append(sample)

    if (i + 1) % 100 == 0: # Log progress
        print(f"Processed {i+1}/{len(hf_dataset)} samples. Adding to FiftyOne dataset...")
        fo_dataset.add_samples(samples)
        samples = [] # Reset batch

# Add any remaining samples
if samples:
    fo_dataset.add_samples(samples)

print(f"Successfully created FiftyOne dataset '{fo_dataset.name}' with {len(fo_dataset)} samples.")
print(f"Images saved in: {fiftyone_image_dir}")

# You can now launch the FiftyOne App to visualize it:
# session = fo.launch_app(fo_dataset)
# print("FiftyOne App launched. If not, open your browser to the URL printed above.")


# ## Launch the FiftyOne visualization app

# In[9]:


session = fo.launch_app(fo_dataset, auto=False)
print(session.url)


# In[10]:


fo_dataset


# ## Create embeddings with MobileNet

# In[10]:


try:
    # 1. Load the MobileNet model from the FiftyOne Model Zoo
    #    Common MobileNet versions include "mobilenet-v2-imagenet-torch"
    #    You can check available models with foz.list_zoo_models()
    mobilenet_model_name = "mobilenet-v2-imagenet-torch" # Example, check zoo for exact names
    model = foz.load_zoo_model(mobilenet_model_name)
    print(f"Successfully loaded model: {mobilenet_model_name}")

    # 2. Define the field name to store the embeddings
    embeddings_field = "mobilenet_embeddings"

    # 3. Compute embeddings
    #    This will iterate through your dataset, pass images through MobileNet,
    #    and store the resulting embedding vector in the 'embeddings_field'.
    fo_dataset.compute_embeddings(model, embeddings_field=embeddings_field, batch_size=16) # Adjust batch_size as needed

    print(f"Embeddings computed and stored in field '{embeddings_field}'.")

    # Verify by checking the first sample (if the dataset is not empty)
    if fo_dataset:
        first_sample = fo_dataset.first()
        if first_sample.has_field(embeddings_field) and first_sample[embeddings_field] is not None:
            print(f"First sample's '{embeddings_field}' shape: {first_sample[embeddings_field].shape}")
        else:
            print(f"First sample does not have the field '{embeddings_field}' or it is None.")
    else:
        print("Dataset is empty, cannot check first sample's embeddings.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure the MobileNet model name is correct and available in the zoo.")
    print("You can list available models using 'fiftyone.zoo.list_zoo_models()'.")


# ## Compute uniqueness of embeddings

# In[12]:


# Compute uniqueness of embeddings
fob.compute_uniqueness(fo_dataset, embeddings=embeddings_field)


# ## Compute representativeness of embeddings

# In[16]:


fob.compute_representativeness(
    fo_dataset,
    embeddings=embeddings_field)


# ## Compute projection to visualize embeddings

# In[17]:


# Visualize with PCA
pca_vis = fob.compute_visualization(
    fo_dataset,
    embeddings=embeddings_field,
    method="pca",
    brain_key="mobilenet_pca"
)
session = fo.launch_app(fo_dataset, auto=False)
print(session.url)


# ## Compute and inspect near duplicates

# In[31]:


duplicate_inspection = fob.compute_near_duplicates(fo_dataset, 
                                                   threshold=0.2,
                                                   embeddings=embeddings_field)
dup_view = duplicate_inspection.duplicates_view()



# In[33]:


session = fo.launch_app(dup_view, auto=False)
print(session.url)


# ## Open vocabulary segmentation with YOLOE

# In[35]:


model = YOLOE("yoloe-11s-seg.pt")

names = ["body of a jaguar", "head of a jaguar"]
model.set_classes(names, model.get_text_pe(names))

fo_dataset.apply_model(model, label_field="yoloe_segmentation")



# In[36]:


session = fo.launch_app(fo_dataset, auto=False)
print(session.url)


# # Further Explorations with FiftyOne
# 
# This notebook has demonstrated several core capabilities of FiftyOne for dataset understanding and preparation. Here are some ideas for further exploration:
# 
# 1.  **Model Evaluation and Error Analysis**:
#     *   Load your model's predictions (classifications, detections, segmentations) back into your FiftyOne dataset.
#     *   Use `fo.evaluate_xxx()` methods (e.g., `evaluate_classifications()`, `evaluate_detections()`) to get detailed performance metrics.
#     *   Visually inspect false positives, false negatives, and misclassifications in the FiftyOne App.
#     *   Filter samples by prediction confidence to find borderline cases.
# 
# 2.  **FiftyOne Brain Methods**:
#     *   **Hardness**: If you have model predictions, use `fo.brain.compute_hardness()` to identify samples that your model struggles with.
#     *   **Mistakenness**: Use `fo.brain.compute_mistakenness()` to find samples that are frequently misclassified.
#     *   **Similarity Search**: After computing embeddings, use `fo.brain.compute_similarity()` to find images similar to a query image or a set of images. This is also great for finding near-duplicates or visually similar instances.
#     *   **Clustering:**: Use the clustering plugin to try different approaches to cluster the embedings. Compare K-means with a
# 
# 3.  **Advanced Querying and Tagging**:
#     *   Construct more complex queries to slice your dataset (e.g., "show me all jaguars with low embedding variance that were misclassified").
#     *   Use FiftyOne's tagging system (`sample.tags.append("needs_review")`) to mark samples for specific actions (e.g., relabeling, removal).
# 
# 4.  **Annotation Integration**:
#     *   If you're doing object detection or segmentation, explore FiftyOne's annotation integrations. You can export data for annotation, import new annotations, and manage annotation workflows.
# 
# 5.  **Dataset Comparison**:
#     *   If you have multiple versions of your dataset or datasets from different sources, use FiftyOne to compare their statistics, label distributions, and embedding spaces.
#     * Compare the embeddings produced with the cropped versions of the images vs the segmented ones. You can use the `segmented_body` split for this or compute your own segmentation using YOLOE or [Mask-RCNN](https://docs.voxel51.com/model_zoo/models.html#mask-rcnn-resnet101-atrous-coco-tf).
# 
# 6.  **Custom Plots and Visualizations**:
#     *   Beyond the built-in plots, you can create custom interactive plots using Plotly (which FiftyOne's plotting is based on) and [attach them to your FiftyOne App session](https://docs.voxel51.com/user_guide/plots.html) for a tailored analysis experience.
# 
# 7.  **Scalability**:
#     *   For very large datasets, explore FiftyOne Enterprise for collaborative features and production-grade scalability.
# 
# 
