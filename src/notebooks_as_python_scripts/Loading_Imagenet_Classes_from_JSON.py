#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests

# The raw URL to the JSON file
url = "https://raw.githubusercontent.com/aihpi/practical-computer-vision-with-pytorch/refs/heads/main/artifacts/imagenet_classes.json"

# Fetch the file
response = requests.get(url)

# Load the JSON content into a dictionary
label_mapping = eval(response.text)

print(label_mapping)


# In[ ]:




