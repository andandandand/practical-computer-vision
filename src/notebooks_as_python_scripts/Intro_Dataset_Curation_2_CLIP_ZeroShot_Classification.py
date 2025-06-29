#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/andandandand/practical-computer-vision/blob/main/notebooks/Intro_to_CLIP_ZeroShot_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # A Tutorial on CLIP (Contrastive Language-Image Pre-training) for Image-Text Similarity and Zero-Shot Classification
# 
# #### Author: Antonio Rueda-Toicen
# **antonio.rueda.toicen 'at' hpi 'dot' de**
# 
# 
# [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
# 
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
# 

# 
# 
# ## An Overview of CLIP
# 
# [CLIP (Contrastive Language–Image Pre-training)](https://arxiv.org/pdf/2103.00020.pdf) is an open source model from OpenAI. Given an image and a set of possible text descriptions, the model predicts the most relevant.
# 
# CLIP overcomes traditional challenges in computer vision. One is the reliance on model outputs limited to specific tasks. The model learns to put representations of similar concepts next to each other. It does this regardless whether they are coming from text or images. This happens through extensive training with text-image pairs scraped from the internet.
# 
# CLIP  adapts to new visual classification tasks without changes in the model's architecture. This is zero-shot inference. It differs from standard image classification models. Those only predict the predefined classes that they were trained on.
# 
# ### Contrastive pre-training
# 
# ![](https://github.com/andandandand/images-for-colab-notebooks/blob/main/clip%20contrastive%20pre-training.png?raw=true)
# 
# CLIP works with two encoders. These are two separate neural networks. One produces embeddings for text and another that produces embeddings for images. Both models “speak the same language” by encoding similar concepts in text and images. These go into embedding vectors of the same size. The text encoder is a BERT-like transformer model or a Continuous Bag of Words (CBOW) model. The Image Encoder can be either a Vision Transformer or a Resnet.
# 
# Given $N$ images and their $N$ matching descriptions, CLIP is trained to predict which of the $NXN$ matchings of image pairs occurred.
# We jointly train an image encoder and a text encoder to maximize cosine similarity of the matching $N$ pairs (main diagonal on the plot), while minimizing
# similarity of tne $N ^ 2 - N$ incorrect pairings.
# The cross entropy of images to text and the cross entropy of text to images is added and averaged. This is called 'symmetric cross entropy' and is the 'contrastive loss' used to train the model. The pseudocde of the method, from the original paper by OpenAI, is shown below.
# 
# ![](https://github.com/andandandand/images-for-colab-notebooks/blob/main/Screenshot%202024-03-04%20at%2009.46.49.png?raw=true)
# 
# 
# With this method, embeddings that represent similar concepts are moved closer together. The effect is what we see in the image below. Embeddings of text and images with high semantic similarity get close to each other. Semantically different text-image pairs are kept further apart.
# 
# ![](https://cdn.sanity.io/images/vr8gru94/production/a54a2f1fa0aeac03748c09df0fdfbb42aadc96b7-2430x1278.png)
# 
# In this notebook, we use CLIP to extract image embeddings. We don't delve further into contrastive pre-training. In a future session, we will learn how to tune a pretrained CLIP model using LORA (Low Rank Adaptation).
# 
# ### Zero-shot image classification
# 
# CLIP can be applied to any visual classification benchmark. We do this by providing the names of the visual categories to classify. Zero-shot prediction allows us generalize on unseen labels. We don't need to specifically train the model to classify them. For example, all ImageNet pretrained models recognize 1000 specific classes. We saw this on previous notebooks. CLIP is not bound by this limitation. With CLIP, we can create new labels 'on the fly'. We pass them through the text encoder. Then we use the similarity between text and image embeddings to produce an output.
# 
# ![](https://github.com/andandandand/images-for-colab-notebooks/blob/main/clip%20zero-shot%20prediction.png?raw=true)
# 
# We explore zero-shot inference with CLIP on this notebook
# 
# ### Limitations and issues
# 
# CLIP has its limitations. It has difficulty with abstract tasks. It's also difficult to generalize to images outside the pre-training dataset. We sometimes need to finetune the model to do good fine-grained classification.
# 
# ![](https://github.com/andandandand/images-for-colab-notebooks/blob/main/coffee-clip.png?raw=true)
# 
# Ethical concerns arise due to the potential for biases coming from the dataset.  There are also privacy and copyright issues. A lot of data from the Internet was used to train this model. Despite these challenges, CLIP's ability highlights the usefulness of applying Internet-scale multimodal datasets.
# 
# The original CLIP model was trained on 400 million image-text pairs. This was done using 256 V100 Nvidia GPUs. This scale is unachievable by most companies and private individuals. It's difficult to create a CLIP model from scratch that performs as well as the pretrained ones by OpenAI or [LAION](https://laion.ai/blog/large-openclip/).
# 
# In this notebook we will explore both the capabilities and limitations of CLIP. We will ask it to produce image embeddings and classifications for paintings. We will explore the Tower of Babel.

# ## The Tower of Babel by Peter Bruegel the Elder (and others)
# 
# 
# Pieter Bruegel the Elder created three paintings on the theme of the [Tower of Babel](https://en.wikipedia.org/wiki/The_Tower_of_Babel_%28Bruegel%29), with two surviving versions housed in Vienna's Kunsthistorisches Museum and Rotterdam's Museum Boijmans Van Beuningen. These paintings depict the biblical story of humanity building a tower to reach the heavens, as described in Genesis. In the story, God is displeased with this development and creates languages (according to Judeochristian tradition there was a single one before this event) in order to sabotage the project and punish humans for their hubris.
# 
# A notable feature of Bruegel's paintings is the architectural similarity of the tower to the [Roman Colosseum](https://en.wikipedia.org/wiki/The_Tower_of_Babel_%28Bruegel%29#Architecture), symbolizing hubris and persecution. This thematic choice reflects the religious tensions of Bruegel's time.
# 
# The theme of the Tower of Babel was popular among other artists as well. Lucas van Valckenborch and Pieter Bruegel the Younger (the Elder's son) also created their own interpretations of the Tower of Babel, further emphasizing its impact on art history.
# 
# We use these paintings and their descriptions to explore CLIP.

# In[ ]:


import skimage.io as io
from PIL import Image
from collections import OrderedDict

url_a = 'https://artinwords.de/wp-content/uploads/Pieter-Bruegel-Turmbau-zu-Babel.jpg'
description_a = 'The (Great) Tower of Babel by Peter Bruegel the Elder'
img_a = Image.fromarray(io.imread(url_a))
img_a


# In[ ]:


url_b = 'https://www.bruegel2018.at/fileadmin/user_upload/Cat_65-HR-Tower-of-Babel.jpg'
description_b =  'The (Little) Tower of Babel by Peter Bruegel the Elder'
img_b = Image.fromarray(io.imread(url_b))
img_b


# In[ ]:


url_c = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/La_Tour_de_Babel%2C_Van_Valckenborch%2C_1594.jpg/1200px-La_Tour_de_Babel%2C_Van_Valckenborch%2C_1594.jpg'
description_c =  "Van Valckenborch's Tower of Babel"
img_c = Image.fromarray(io.imread(url_c))
img_c


# In[ ]:


url_d = 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Pieter_Bruegel_d._%C3%84._109.jpg/1600px-Pieter_Bruegel_d._%C3%84._109.jpg'
description_d = "King and Entourage - Detail from the Great Tower of Babel"
img_d = Image.fromarray(io.imread(url_d))
img_d


# In[ ]:


url_e = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/The_Tower_of_Babel%2C_oil_on_panel_painting_by_Pieter_Brueghel_the_Younger.jpg/1476px-The_Tower_of_Babel%2C_oil_on_panel_painting_by_Pieter_Brueghel_the_Younger.jpg?20161130233735'
description_e =  "Tower of Babel by Peter Brugel the Younger"
img_e = Image.fromarray(io.imread(url_e))
img_e


# In[ ]:


url_f = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Hieronymus_Cock_-_Kolosseum_1551_beschnitten.jpg/1600px-Hieronymus_Cock_-_Kolosseum_1551_beschnitten.jpg'
description_f = "A sketch of the Roman Colosseum by Hieronymus Cock"
img_f = Image.fromarray(io.imread(url_f))
img_f


# In[ ]:


url_g = 'https://github.com/andandandand/images-for-colab-notebooks/blob/main/babellora02.png?raw=true'
description_g = "Bruegel's Tower of Babel, generated by Stable Diffusion XL 1.0"
img_g = Image.fromarray(io.imread(url_g))
img_g


# ## Importing CLIP from HuggingFace

# In[ ]:


from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
import requests


# In[ ]:


# We can inspect the model's architecture, which is PyTorch code.
# We have downloaded both the architecture and the model's weights.
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
model


# The `transformers` library from Huggingface allows us to use many pre-trained models.
# 
# `CLIPModel.from_pretrained('openai/clip-vit-base-patch32')` is a function call that loads a pre-trained CLIP model, the `clip-vit-base-patch32` version.
# 
# Here's what happens during this function call:
# 
# 1. Model Architecture Loading: The `CLIPModel` class represents the CLIP model architecture. When we call `from_pretrained()`, it initializes a model with the architecture defined for CLIP.
# 
# 2. Pre-trained Weights: The string `'openai/clip-vit-base-patch32'`  is a set of weights. These weights are in the Hugging Face model hub. OpenAI produced these with the CLIP learning method on a large dataset of images and their descriptions.
# 
# 3. Vision Transformer Variant: `vit-base-patch32` indicates that the model uses the [Vision Transformer (ViT)](https://paperswithcode.com/method/vision-transformer) architecture. The input images are divided into patches of size 32x32 pixels before processing by the transformer.
# We could also choose a [ResNet (convolutional neural network)](https://paperswithcode.com/method/resnet) instead.
# 
# 4. Downloading and Caching: If this is the first time we're using this model in the running working space, the weights are downloaded from the Hugging Face model hub and cached locally. The following usage of `from_pretrained()` will use the local cache, without downloading the weights again.
# 
# 5. Instantiation and Readiness for Inference: After the model weights load, the CLIP model is ready for inference. You can then process images and text to extract text of image features (vector embeddings) or perform zero-shot classification tasks.

# In[ ]:


processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
processor


# The `CLIPProcessor` prepares data for CLIP. The model requires both images and text to be in a specific format before being fed to the encoding networks. Here's how it works:
# 
# 1. **Resizing and Normalization**: The processor takes an image input. It then resizes it to the dimensions expected by the model (e.g., 224x224 pixels). It then normalizes the image by scaling pixel values to a range that the model was trained on, typically [0, 1] or [-1, 1]. It then  aligns it with the color channel means and standard deviations that the model expects. For this model these are the mean and standard deviations fof the RGB channels on the [CLIP training dataset](https://arxiv.org/pdf/2103.00020.pdf). CLIP was trained from scratch by OpenAI without using ImageNet weights for the visual encoder or other weights for the text encoder.
# 
# 2. **Tokenization**: For the text inputs, the processor [tokenizes the sentences](https://claritynlp.readthedocs.io/en/latest/developer_guide/algorithms/sentence_tokenization.html). We convert the text to tokens (often words or syllables) that are represented by numerical IDs. These IDs correspond to entries in the model's vocabulary.
# 
# 3. **Padding and Attention Mask**: The processor pads the token sequences. They are made to be the same length for batch processing. It also creates attention masks that allow the model to ignore padding tokens during processing.
# 
# 4. **Conversion to PyTorch Tensors**: The processor converts the processed image and text data into PyTorch tensors. Tensors are multi-dimensional arrays suitable for input into the model.
# 
# 5. **Return Tensors**: The processed tensors return in a format that can be fed into the CLIP model. Now we can produce classifications or embeddings from them.
# 
# 

# ## Producing Text Embeddings

# In[ ]:


# We create a list of descriptions to pass to the processor
descriptions = [description_a, description_b, description_c, description_d,
                description_e, description_f, description_g]
descriptions


# In[ ]:


inputs = processor(descriptions, return_tensors="pt",
                   padding=True, truncation=True)


# We use torch.no_grad() to avoid having to call .detach() on the tensor
with torch.no_grad():
    text_embeddings = model.get_text_features(**inputs)


# In[ ]:


text_embeddings.shape


# In[ ]:


inputs = processor(images=[img_a, img_b, img_c, img_d, img_e, img_f, img_g],
                   return_tensors="pt")
inputs.keys()


# The line `inputs = processor(images=img_a, return_tensors="pt")` is using the `CLIPProcessor` to preprocess an image so that it can be inputted into the CLIP model for inference. Here is a breakdown of what each part of this line of code does:
# 
# 1. **processor(images=img_a)**: This part of the code calls the `processor` with the image `img_a` as an argument. The `images` parameter is where we pass the image or images we want to process. The processor will handle the necessary transformations of this image so that it is in the correct format for the CLIP model.
# 
# 2. **return_tensors="pt"**: This argument tells the processor that we want the output to be PyTorch tensors. The `"pt"` stands for PyTorch. If we were working with TensorFlow, for example, we might use `"tf"` to get TensorFlow tensors instead.
# 
# 3. **inputs**: This variable is being assigned the output of the processor. This output will be a dictionary containing everything the model needs to run its predictions. This typically includes the processed pixel values of the image, now as a tensor, and an attention mask indicating which parts of the tensor are actual data and which parts are padding.
# 
# The processor's output, which is now stored in `inputs`, can be directly fed into the CLIP model to obtain embeddings or perform inference. It simplifies the process by abstracting the preprocessing steps such as resizing, normalization, and conversion to tensors, which are all necessary to prepare the image data for the model.

# In[ ]:


import numpy as np
np.array(img_a).shape


# In[ ]:


index=0
inputs['pixel_values'][0].shape


# In[ ]:


np.array(img_a).min(), np.array(img_a).max()


# In[ ]:


inputs['pixel_values'][0].min(), inputs['pixel_values'][0].max()


# ## Effects of Image Resizing and Normalization
# 
# The `CLIPProcessor` from the Hugging Face Transformers library preprocesses images in a specific way for use with CLIP models. The observed range of values outside of -1 and 1 after processing with `CLIPProcessor` might be unexpected when considering the usual normalization technique, which aims to scale pixel values between -1 and 1 based on the mean and standard deviation of the data. However, the range can exceed these bounds due to the specifics of how we are doing preprocessing.
# 
# When an image is processed by `CLIPProcessor`, several steps are typically applied:
# 
# 1. **Resizing and Cropping:** The image is resized and cropped to match the input size expected by the CLIP model. In our example, we turn our image of arbitrary size and aspect ratio into a square tensor of size (224, 224).
# 
# 2. **Normalization:** The pixel values, which are initially in the range [0, 255], are normalized. The normalization is done using pre-defined mean and standard deviation values. The formula used is:
# 
#    $$
#    \text{normalized value} = \frac{\text{value} - \text{mean from training set}}{\text{std from training set}}
#   $$
#   
#    For CLIP models, the mean and standard deviation are usually set to values that are specific to the dataset on which CLIP was trained. Each R,G,B channel from the training set has its own mean and standard deviation.
# 
# The range of normalized values exceeding -1 and 1 can occur if the original pixel values of the image significantly deviate from the expected mean, even after normalization. This deviation can result from the specific mean and standard deviation values used in the normalization step, which come from the distribution of the dataset on which CLIP was trained.
# 
# The normalization step assumes that the input pixel values are distributed in a certain way, usually centered around the dataset mean. If an image has pixel values that are not well represented by the dataset mean and standard deviation (for example, if the image is significantly brighter or darker than the average image in the dataset), the normalized values can fall outside the expected range of -1 to 1.
# 
# In summary, the observed range of values outside of -1 and 1 after preprocessing with `CLIPProcessor` is a result of how the normalization step interacts with the specific characteristics of your input images relative to the expected distribution of the dataset used to train the CLIP model. This behavior is typically not an issue for the model, as CLIP and similar neural networks are generally robust to variations in input ranges.

# In[ ]:


# @title How the model 'views' the data {run:'auto'}
slider_value = 0  # @param {type: "slider", min: 0, max: 5}
import matplotlib.pyplot as plt
plt.imshow(inputs['pixel_values'][slider_value].permute(1,2,0))#.view(224, 224, 3));


# ## Producing image embeddings

# In[ ]:


image_embeddings = model.get_image_features(**inputs).detach()
image_embeddings


# In[ ]:


image_embeddings.shape


# The line `outputs = model.get_image_features(**inputs)` is invoking a method of the `CLIPModel` instance to obtain image features from the preprocessed inputs.
# This line is where the actual computation happens to transform your image into a fixed-size vector (embedding) that captures the visual essence of the image, as learned by the model during its training on a diverse set of images and text descriptions.
# 
# Here's a step-by-step explanation:
# 
# - **model**: This is the instance of the `CLIPModel` that we have loaded using `CLIPModel.from_pretrained('openai/clip-vit-base-patch32')`. It encapsulates the pre-trained CLIP model.
# 
# - **get_image_features**: This method of the `CLIPModel` class is used to get the image features from the input data. Image features are essentially a vector of numbers that represent the contents of the image in a way that can be understood and used by machine learning models.
# 
# - **(**inputs**)**: The double asterisk `**` is used to unpack the `inputs` dictionary into keyword arguments. This means that if `inputs` contains, `{'pixel_values':tensor, 'attention_mask': tensor}`, calling `**inputs` would be like passing `pixel_values=tensor, attention_mask=tensor` directly to the function.
# 
# - **outputs**: The variable `outputs` is being assigned the result of `get_image_features`. After this line executes, `outputs` will contain the image embeddings, which are the feature representations of the input image. These embeddings capture the visual information in a form that can be used for comparison with other images, classification, and other tasks that the CLIP model is capable of.
# 
# 

# In[ ]:


image_embeddings.shape


# In[ ]:


image_embeddings[0].shape


# ## `torch.nn.CosineSimilarity`
# 
# 
# 
# * `dim`: This parameter specifies the dimension along which cosine similarity is computed. `dim=0` means that the similarity will be computed along the first dimension (i.e., the rows if we think of a 2D tensor as a matrix).
# 
# * `eps`: This is a small value added to the denominator for numerical stability. In the code eps=1e-6, it prevents division by zero when normalizing vectors. This is especially useful when dealing with very small values in the vectors.
# 
# $$
# \text{cosine similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
# $$
# 

# In[ ]:


cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)


# In[ ]:


def show_pair(imag_a, imag_b):
  plt.subplot(121)
  plt.imshow(np.array(imag_a))
  plt.axis("off")

  plt.subplot(122)
  plt.imshow(np.array(imag_b))
  plt.axis("off")


# In[ ]:


# We create two lists to go through the images and descriptions
images = [img_a, img_b, img_c, img_d, img_e, img_f, img_g]
descriptions = [description_a, description_b, description_c, description_d,
                description_e, description_f, description_g]


# ## Evaluating the similarity of image embeddings

# In[ ]:


# @title {run:'auto'}

slider_value_1 = 0  # @param {type: "slider", min: 0, max: 7}
slider_value_2 = 1  # @param {type: "slider", min: 0, max: 7}


print(f'Cosine similarity = {cosine_similarity(image_embeddings[slider_value_1], image_embeddings[slider_value_2]):.2f}')
show_pair(images[slider_value_1], images[slider_value_2])


# ## Evaluating the similarity of text embeddings

# In[ ]:


# @title {run:'auto'}

slider_value_1 = 0  # @param {type: "slider", min: 0, max: 5}
slider_value_2 = 1  # @param {type: "slider", min: 0, max: 5}


print(f"""Cosine similarity = {cosine_similarity(text_embeddings[slider_value_1],
                                               text_embeddings[slider_value_2]):.2f}""")
print(f"""First description: {descriptions[slider_value_1]}\nSecond description: {descriptions[slider_value_2]}""")


# ## Evaluating the text-image similarity of embeddings

# In[ ]:


# @title {run:'auto'}

slider_value_1 = 0  # @param {type: "slider", min: 0, max: 5}
slider_value_2 = 0  # @param {type: "slider", min: 0, max: 5}


print(f"""Cosine similarity = {cosine_similarity(text_embeddings[slider_value_1],
                                                image_embeddings[slider_value_2]):.2f}""")
print(f"""Description: {descriptions[slider_value_1]}""")
images[slider_value_2]


# In[ ]:


text_embeddings.shape, image_embeddings.shape


# In[ ]:


# Notice that the embeddings are not normalized
text_embeddings.max(), image_embeddings.max()


# In[ ]:


# We normalize embeddings (the cosine_similarity function did this for us before)
image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
text_embeddings.max(), image_embeddings.max()


# In[ ]:


similarity = text_embeddings @ image_embeddings.T
similarity


# In[ ]:


count = len(descriptions)

plt.figure(figsize=(20, 14))
plt.imshow(similarity, vmin=0.1, vmax=0.3)
plt.yticks(range(count), descriptions, fontsize=18)
plt.xticks([])
for i, image in enumerate(images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
  plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image embeddings", size=20);


# ## Running CLIP as a zero-shot classifier

# In[ ]:


text = ['a photograph',
        'a painting',
        'a cartoon',
        'a comic',
        'a newspaper',
        'live people standing in front of a painting',
        'a painting by Peter Bruegel the elder',
        'a painting by Peter Bruegel the younger',
        'a painting by Picasso',
        'a painting by Dali',
        'a painting by PJ Crook',
        'a painting by Van Valckenborch',
        'a sketch of the Roman Colosseum',
        'The Great Tower of Babel by Peter Bruegel the Elder',
        'The  Tower of Babel by Peter Bruegel the Younger',
        'The Little Tower of Babel by Peter Bruegel the Elder',
        'The Tower of Babel by Van Valckenborch',
        'The Tower of Babel generated by Stable Diffusion'
        ]

inputs = processor(text=text,
                   images=[img_a, img_b, img_c, img_d, img_e, img_f],
                   return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
temperature = 1.0
outputs.logits_per_image.shape


# 

# ### Understanding softmax's temperature parameter
# 
# In machine learning, especially in the context of neural networks, we often use the softmax function for multi-class classification problems. It transforms a vector of raw scores (logits) from the model into probabilities. This is achieved by taking the exponential of each element and then normalizing these values by dividing by the sum of all these exponentials. This process ensures that the output values fall within the range (0, 1) and sum up to 1, making them interpretable as probabilities.
# 
# The temperature parameter $(T$) of the softmax function is a hyperparameter that influences the "sharpness" of the output probability distribution. The modified softmax function with temperature is defined as:
# 
# $$
# \text{Softmax}(\mathbf{z_i}) = \frac{e^{\frac{z_i}{T}}}{\sum_{j} e^{\frac{z_j}{T}}}
# $$
# 
# where:
# - $\mathbf{z}$ is the input vector containing the raw class scores from the model,
# - $z_i$ is the score for class $i$,
# - $T$ is the temperature parameter, and
# - $i$ ranges over all possible classes.
# 
# ### Impact of Temperature $T$
# 
# - **$T = 1$:** This represents the standard softmax function without modification, keeping the differences in the scores unchanged.
# 
# - **$T > 1$:** Increasing the temperature makes the softmax function "softer", leading to a more uniform probability distribution. Higher temperatures result in smaller differences between the largest and smallest probabilities, promoting exploration in models used in reinforcement learning or in sequence generation where output diversity is desired.
# 
# - **$T < 1$:** Decreasing the temperature makes the softmax function "sharper", amplifying the differences between the scores. A lower temperature results in a greater disparity between the highest probability and the rest, enhancing the model's confidence (though potentially increasing the risk of misplaced confidence). This sharper distribution can be beneficial in scenarios requiring more decisive actions.
# 
# ### Example
# 
# Let's consider a vector of logits $\mathbf{z} = [2, 1, 0.1]$ and apply the softmax function with different temperatures:
# 
# - **For $T = 1$:**
#   - The differences between the scores are maintained as originally presented.
# - **For $T > 1$, say $T = 2$:**
#   - The softmax output makes the scores closer to each other, leading to a more uniform probability distribution.
# - **For $T < 1$, say $T = 0.5$:**
#   - The highest score is significantly more emphasized in the probability distribution, making it peakier.
# 
# By adjusting the temperature parameter, we can control the level of confidence in the predictions our model makes.

# In[ ]:


# Probabilities at different temperatures @title {run:'auto'}

slider_value = 0  # @param {type: "slider", min: 0, max: 5}
temperature = 0.7  # @param {type: "slider", min: 0, max: 2, step:0.1}

# Creating a subplot with an image in the first row and the histogram in the second row
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

axes[0].imshow(np.array(images[slider_value]))
axes[0].axis('off')  # Turning off the axis for the image
axes[0].set_title(descriptions[slider_value])

probs = (logits_per_image/temperature).softmax(dim=1)

# Creating the horizontal bar plot in the second subplot
axes[1].barh(text, probs[slider_value].detach(), color='skyblue')
axes[1].set_xlim(0, 1)  # Setting the x-axis limit from 0 to 1

axes[1].set_xlabel('Probabilities')
axes[1].set_title('Probabilities of Different Labels')

plt.tight_layout()
plt.show()


# ## Now try it on a cat image! 🐱

# In[ ]:


from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
cats_img = Image.open(requests.get(url, stream=True).raw)
cats_img


# In[ ]:


# Notice what happens with the output probabilities
# when we make the labels more specific
cat_text = ['a cat',
            'a dog',
            'two cats',
           'two cats lying on a sofa',
           'two cats lying on a sofa next to two tv controllers',
           'two dogs lying on a sofa next to two tv controllers'
           ]

inputs = processor(text=cat_text,
                   images=[cats_img],
                   return_tensors="pt", padding=True)
cat_outputs = model(**inputs)
cat_logits_per_image = cat_outputs.logits_per_image


# In[ ]:


# @title Probabilities at different temperatures {run:'auto'}

temperature = 0.01  # @param {type: "slider", min: 0.01, max: 2.0, step:0.01}

# Creating a subplot with an image in the first row and the histogram in the second row
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

axes[0].imshow(np.array(cats_img))
axes[0].axis('off')  # Turning off the axis for the image

cat_probs = (cat_logits_per_image/temperature).softmax(dim=1)

# Creating the horizontal bar plot in the second subplot
axes[1].barh(cat_text, cat_probs.detach().numpy().flatten(), color='skyblue')
axes[1].set_xlim(0, 1)  # Setting the x-axis limit from 0 to 1

axes[1].set_xlabel('Probabilities')
axes[1].set_title('Probabilities of Different Labels')

plt.tight_layout()
plt.show()


# ## What about the lion-boar? 🦁

# In[ ]:


from PIL import Image
import requests

url = "https://github.com/andandandand/images-for-colab-notebooks/blob/main/Screenshot%202024-02-13%20at%2011.44.27.png?raw=true"
lion_boar_img = Image.open(requests.get(url, stream=True).raw)
lion_boar_img


# In[ ]:


lion_boar_text = ['a lion',
                  'a puma',
                  'a guy dressed like a puma',
                  'a boar',
                  'a female lion',
                  'a male lion',
                  'a lion cub'
                ]

inputs = processor(text=lion_boar_text,
                   images=[lion_boar_img],
                   return_tensors="pt", padding=True)
lion_boar_outputs = model(**inputs)
lion_boar_logits_per_image = lion_boar_outputs.logits_per_image


# In[ ]:


# Probabilities at different temperatures @title {run:'auto'}

temperature = 1.71  # @param {type: "slider", min: 0.01, max: 2, step:0.1}

# Creating a subplot with an image in the first row and the histogram in the second row
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

axes[0].imshow(np.array(lion_boar_img))
axes[0].axis('off')  # Turning off the axis for the image

lion_boar_probs = (lion_boar_logits_per_image/temperature).softmax(dim=1)

# Creating the horizontal bar plot in the second subplot
axes[1].barh(lion_boar_text, lion_boar_probs.detach().numpy().flatten(), color='skyblue')
axes[1].set_xlim(0, 1)  # Setting the x-axis limit from 0 to 1

axes[1].set_xlabel('Probabilities')
axes[1].set_title('Probabilities of Different Labels')

plt.tight_layout()
plt.show()


# Please continue the exploration of class attribution [in this notebook](https://colab.research.google.com/drive/1oItQTATutrRrFSvXv2ThPdr4vy1ogRWN?usp=sharing) (Requires GPU usage)

# ## References
# 
# * [OpenAI's CLIP Announcement](https://openai.com/research/clip)
# 
# * [Learning Transferable Visual Models From Natural Language Supervision
# ](https://arxiv.org/abs/2103.00020)
# 
# * [Pinecone's CLIP Tutorial](https://www.pinecone.io/learn/series/image-search/clip/)
# 
# * [Wikipedia's article on Peter Bruegel's Tower of Babel](https://en.wikipedia.org/wiki/The_Tower_of_Babel_%28Bruegel%29)
# 
# * [List of Paintings by Peter Bruegel the Elder](https://en.wikipedia.org/wiki/List_of_paintings_by_Pieter_Bruegel_the_Elder)
# 
# * [CLIP on HuggingFace](https://huggingface.co/docs/transformers/model_doc/clip)
