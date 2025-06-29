#!/usr/bin/env python
# coding: utf-8

# # Generating Artwork Images with Stable Diffusion and Inspecting the Output with Embeddings

# You will want to use a GPU to run this environment.In Google Colab go to Runtime -> Change Runtime Type and select one of the available Nvidia GPUs.

# In[31]:


# Install and fix required packages
get_ipython().system('pip install transformers==4.51.3 accelerate==1.7.0 scipy==1.15.3 safetensors==0.5.3 diffusers==0.33.1 fiftyone==1.5.2 open-clip-torch==2.32.0 > /dev/null')


# In[32]:


# Check that we have an environment with a GPU
# Go to Runtime -> Change Runtime Type to select the T4 GPU
# Without it, this cell will fail
get_ipython().system('nvidia-smi')


# #Setup input + output

# In[33]:


#set params & select model

model = "stabilityai/stable-diffusion-xl-base-1.0"


# In[34]:


#set up folder configuration
import os
from google.colab import files, drive

drive.mount('/content/drive')

PROJECT_NAME = 'artstyle5'

BASE_FOLDER = os.path.join(os.getcwd(), 'drive', 'MyDrive','Colab Notebooks')
OUTPUT_IMAGE_FOLDER = os.path.join(BASE_FOLDER, 'Projects', PROJECT_NAME)

if not os.path.exists(OUTPUT_IMAGE_FOLDER):
    print(f'{OUTPUT_IMAGE_FOLDER} was not present and was created')
    os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)


# # Load SDXL pipeline
# 

# In[35]:


#import libraries
from diffusers import StableDiffusionXLPipeline #to use the refiner import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import torch


# In[36]:


#Load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
  model,
  torch_dtype=torch.float16,
  use_safetensors=True,
  variant="fp16"
)
pipe.to("cuda")

print(f'{model} loaded')


# #Generate images

# In[37]:


#gets the index of the last generated image
def get_last_index(imgfolder): #os.listdir(OUTPUT_IMAGE_FOLDER)
  outfiles = [file for file in os.listdir(imgfolder) if os.path.isfile(os.path.join(imgfolder,file))]
  if len(outfiles)>0:
    try:
      img_num = sorted([int(img.split('.')[0]) for img in os.listdir(imgfolder) if img.split('.')[0].isdigit()])[-1] #gets last image in the output folder (highest number)
      return img_num
    except:
      return 0
  else:
    return 0

get_last_index(OUTPUT_IMAGE_FOLDER)


# In[ ]:


import random
import pandas as pd

art_styles = ['Rembrandt',  'Magritte', 'Monet', 'Klimt', 'Renoir', 'Peter Bruegel', 'Chagall',
              'Hokusai', 'Hiroshige','Steampunk', 'Art Deco', 'Art Nouveau', 'Paul Gauguin',
              'Banksy', 'Goya', 'Van Gogh', 'Leonardo da Vinci', 'Kandinsky', 'Mondrian']
art_types = ['beautiful lansdcape', 'medieval town', 'sunset view from a park', 'desert',
             'modern city at night, Berlin', 'Potsdamer Platz', 'arctic landscape with aurora borealis',
             'portrait of mother and child', 'musical instruments', 'jazz band playing music', 'guitar player',
             'skyscrappers', 'erupting volcano', 'sunny beach and sea', 'portrait of a woman', 'view of Berlin',
             'spaeti shop in Berlin', 'Alexanderplatz', 'cafe at night',
             'desert', 'jungle', 'painting of mechanical device', 'supercomputer','river bank', 'view of a city with bridge',
             'still nature', 'wolf', 'a view of the Spree', 'crows in Berlin', 'crows and foxes in Berlin Tiergarten']

img_num = get_last_index(OUTPUT_IMAGE_FOLDER)

STARTING_RANGE = img_num+1 if img_num!=0 else 0 #starts image index after last index or 0 if no images found

ADD_RANGE = 200 #how many images to generate

prompt_save = [] #this keeps the info for the experiment sheet

for i, seed in enumerate(range(STARTING_RANGE, STARTING_RANGE+ADD_RANGE)):
    generator = torch.Generator("cuda").manual_seed(seed)
    art_style = random.choice(art_styles)
    art_type = random.choice(art_types)
    prompt = f'{art_type} in the style of {art_style}'
    print(f'Processing image {i+1} / {ADD_RANGE} | seed: {seed} | prompt: {prompt}')
    negative_prompt = None
    image = pipe(prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                num_inference_steps=25)
    image = image.images[0]
    savefilename = f'{seed}_{prompt}.png'
    image.save(os.path.join(OUTPUT_IMAGE_FOLDER, savefilename))
    prompt_save.append([savefilename, seed, art_style, prompt, negative_prompt])


# In[ ]:


from pathlib import Path
prompt_file = pd.DataFrame(prompt_save, columns=['filename', 'seed','art_style', 'prompt', 'negative_prompt'])
prompt_file.to_csv(Path(OUTPUT_IMAGE_FOLDER).parent / f'trial_up_to_{seed}.csv', index=False)
prompt_file


# In[ ]:


import fiftyone as fo

# Create a FiftyOne dataset
if fo.dataset_exists("stable-diffusion-xl-images"):
  fo.delete_dataset("stable-diffusion-xl-images")
dataset = fo.Dataset(name="stable-diffusion-xl-images")

# Add the art_style, prompt, and seed fields to the dataset schema
dataset.add_samples(
    [
        fo.Sample(
            filepath=os.path.join(OUTPUT_IMAGE_FOLDER, row['filename']),
            tags=[],  # Add tags if you have any
            metadata=None,  # Add metadata if you have any
            art_style=row['art_style'],
            prompt=row['prompt'],
            seed=row['seed'],
        )
        for index, row in prompt_file.iterrows()
    ]
)
dataset.persistent = True
dataset.save()


# In[ ]:


session = fo.launch_app(dataset, auto=False)
print(session.url)


# # Computing and visualizing embeddings with CLIP
# 
# Embeddings are powerful numerical representations of unstructured data like images. Models like CLIP (Contrastive Language-Image Pre-Training) can generate embeddings that capture semantic meaning.
# 
# A CLIP-based model was used to guide the generation process of these images. It's how we connect the text prompts with the generated output
# 
#  By visualizing these embeddings in a lower-dimensional space (e.g., using PCA, UMAP or t-SNE), we can explore similarities and patterns in our dataset.
# 
# FiftyOne's Brain provides tools to compute and visualize embeddings. We'll use the `compute_visualization()` method with a CLIP model.

# ![](https://github.com/andandandand/practical-computer-vision/blob/main/images/generated_artwork.png?raw=true)

# In[ ]:


import fiftyone.brain as fob
import fiftyone.zoo as foz


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

# In[ ]:


# Load a model with embeddings
model = foz.load_zoo_model("clip-vit-base32-torch")

# Generate embeddings for each sample
embeddings = dataset.compute_embeddings(model)


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
# ![]()
# 
# To access this view click on the `+` symbol next to samples, then `Curate -> Embeddings` and Select the brain key that we used to compute the projection. Then `"Split Horizontally"` to access the view of the embeddings next to the images. You can then use the lasso selection tool to highlight portions of the dataset.

# In[ ]:


# Compute visualization in two dimensions
two_dim_projection = fob.compute_visualization(
    dataset, embeddings=embeddings, seed=51, brain_key="embeddings_2d", method="pca"
)


# In[ ]:


# Compute uniqueness of embeddings
fob.compute_uniqueness(dataset, embeddings=embeddings)


# In[ ]:


# Launch the session
session = fo.launch_app(dataset, auto=False)
print(f"The app is available at {session.url}")


# ## Acknowledgements
# 
# Many thanks to [Lucian Renita](https://www.linkedin.com/in/lucian-renitsa/?originalSubdomain=de) for his help and feedback producing this notebook.

# In[ ]:




