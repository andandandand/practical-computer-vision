#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/andandandand/practical-computer-vision/blob/main/notebooks/Training_a_Multilayer_Perceptron_for_Image_based_Regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Tutorial - Fundamentals of Neural Network Training with Image-based Regression
# 
# #### Author: Antonio Rueda-Toicen
# 
# **antonio.rueda.toicen 'at' hpi 'dot' de**
# 
# 
# [![Creative Commons License](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
# 
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
# 
# 
# The pixels on an image each represent an input unit on the input layer. If the input image has 224x224 pixels, each them is a unit on the input (such network requires 50176 input units).
# 
# We have as many units on the output layer as values that we want to predict. If we are to predict only single value (e.g. price, age, height) given an image, we have a single output unit.
# 
# The number of hidden layers and the amount of units that we put in them are decisions that we make as designers of the model. We inform these decisions based on performance on a validation set for our problem. On the validation set, we might try different number of hidden layers and different number of units. We choose the combination that gives us the best performance based on the loss function or another metric related to our problem.
# 
# <img src="https://raw.githubusercontent.com/andandandand/images-for-colab-notebooks/refs/heads/main/Screenshot%202025-03-07%20at%2011.43.55.png" width="300" height="300">
# 
# [Diagram of a multilayer perceptron](https://alexlenail.me/NN-SVG/index.html) where the width of edges is proportional to their weights and their colors blue when negative, red when positive.
# 
# ## What does it mean to "train" a neural network?
# 
# $$w_{ij} = w_{ij} - \left(\text{learning rate} * \frac{dL}{dw_{ij}}\right)$$
# 
# Using the gradient of the loss function to update the weights. In this notebook we go deep into how this is implemented in pure PyTorch.
# 
# As practitioners, in 99.99% of use cases we **don't implement the weight update rule by hand**. We usually pass the parameters of the model to an instance of `torch.optim.optimizer` and then call `optimizer.step()`. In this notebook we explore how this process works by making a neural network learn the price of a car by training and evaluating on a single image. In the next notebook, we will explore **the importance of validation and test sets** to determine whether our network is [overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting#:~:text=Overfitting%20means%20creating%20a%20model,worthless%20in%20the%20real%20world.) or [underfitting](https://www.ibm.com/think/topics/underfitting) to the data.
# 
# 
# ## Memorizing the data
# 
# In this notebook we are training a model using *a single image*,  forcing it to memorize the target value.
# 
# It's important to understand that, in the context most neural network training runs, generalization capabilities only arise when having a rich enough dataset that doesn't represent [spurious correlations](https://www.tylervigen.com/spurious-correlations) between between input and target variables. "Deep" learning only refers to having a "large" number of hidden layers, not finding the deepest secrets of the universe 😜.
# 
# "[Garbage in, garbage out](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out)" remains true)🗑️.
# 
# 
# ## Measuring error with a loss function
# 
# Any differentiable function where we can compute gradients with respect to the weights can be used to update them
# 
# $$L = \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
# 
# 
# ## Parameters vs Hyperparameters
# 
# * Parameters = weights of the model
# * Hyperparameters = network architecture, training setup, e.g. the number of hidden layers, the number of units on each, the learning rate
# 
# The learning rate is a particularly important and sensitive parameter that we explore in this notebook.
# 
# 

# ## Understanding the update rule on a network's weights

# In[38]:


import skimage.io as io
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt


# In[39]:


corvette_array = io.imread("https://github.com/andandandand/images-for-colab-notebooks/blob/main/1964-chevrolet-corvette-stingray.jpeg?raw=true")
# original image dimensions
corvette_array.shape


# In[40]:


Image.fromarray(corvette_array)


# In[41]:


# This is the true price which we are trying to make the network learn
y = torch.tensor(83999.)


# In[42]:


transform_to_square_tensor = transforms.Compose([
    transforms.ToImage(),
    # We first resize the image to Height=Width=224 to reduce the number of parameters
    transforms.Resize((224, 224)),
    transforms.ToDtype(torch.float32, scale=True)
])


# In[43]:


image_tensor = transform_to_square_tensor(corvette_array)
C, H, W = image_tensor.shape
print(C, H, W)


# In[44]:


plt.imshow(image_tensor.permute(1, 2 ,0));


# ## Defining the Architecture of a Multi-layer Perceptron

# In[45]:


# Number of units on the hidden layer
p = 100
# We use nn.Sequential to concatenate PyTorch modules in order
# nn.Linear are modules for fully connected layers.
# The output of the first linear layer is sent to a ReLU function to introduce non-linearity.
multilayer_perceptron=nn.Sequential(
                      nn.Linear(in_features=C * H * W,
                                out_features=p,
                                bias=False),
                      nn.ReLU(),
                      nn.Linear(in_features=p,
                                out_features=1,
                                bias=False),
)
multilayer_perceptron


# In[46]:


# What is inside the multilayer_perceptron is a sequence of nn.Module objects
# Each can take in inputs of the previous module and give outputs to the next
# The Sequential object can be accessed by its indices
len(multilayer_perceptron)


# In[47]:


# Our loss function
def absolute_error(y, y_hat):
    return torch.abs(y - y_hat)


# In[48]:


# We can acccess the weights of every fully connected layer
# These are the randomly initialized weights of the first layer
print(multilayer_perceptron[0].weight.shape)
multilayer_perceptron[0].weight


# In[49]:


# Notice that the dimension of the output appears first
print(multilayer_perceptron[2].weight.shape)
multilayer_perceptron[2].weight


# In[50]:


# Notice that image_tensor has by default requires_grad = False
# as the values of the image tensor (the input) do not need to be updated
image_tensor.requires_grad


# In[51]:


# nn.Linear requires its input to be flattened, the image tensor as we have it now
# has to be reshaped
image_tensor.shape


# In[52]:


# We flatten the tensor to make matrix multiplication work with the
# first layer of weights in the fully connected layer
image_tensor = image_tensor.flatten()
image_tensor.shape


# In[53]:


# We add a batch dimension, as it is a requirement for the input to instances of nn.Module
# You can try removing the unsqueeze(0) call before passing the tensor to the model and interpreting the error
image_tensor = image_tensor.unsqueeze(0)
image_tensor.shape


# In[54]:


# We obtain our prediction
# grad_fn shows the last operation that was performed on the tensor
# MmBackward0 indicates that the tensor was created as a result
# of a matrix multiplication operation during the forward pass of the model.
# It is marked with Backward because it is meant to be used during the computation of the gradients with backpropagation.
# PyTorch automatically generates these backward functions and assigns them names like "AddBackward0", "MulBackward0", "MeanBackward1" etc.
# The number at the end is just to make the name unique.
y_hat = multilayer_perceptron(image_tensor)
y_hat


# In[37]:


# We see that before training, the neural network
# 'believes' that the car is worth less than 1 USD
y, y_hat


# In[18]:


# After computing the loss value grad_fn is updated
# PyTorch's autograd is tracking every operation done
# on outputs that involved the weights
loss_1 = absolute_error(y, y_hat)
loss_1


# In[19]:


# Calling backward on the loss value will create gradients on the nn.Linear layers
loss_1.backward()


# In[20]:


# Notice that the gradient tensor has the same exact dimensions as the weight tensor
assert multilayer_perceptron[0].weight.grad.shape == multilayer_perceptron[0].weight.shape
multilayer_perceptron[0].weight.grad


# ## Implementing the Weight Update Rule
# 
# The weight update rule is the core of gradient descent, here we implement it in PyTorch.
# 
# $$W_{ij}^{(t+1)} = W_{ij}^{(t)} - \text{learning_rate} \cdot \frac{\partial L}{\partial W_{ij}}$$
# 
# 

# In[21]:


# Here we implement the weight update rule
def weight_update_rule(weight, grad, learning_rate):
    return nn.Parameter((weight - learning_rate * grad))


# In[ ]:


learning_rate = 0.1
new_weights_0 = weight_update_rule(multilayer_perceptron[0].weight,
                            multilayer_perceptron[0].weight.grad,
                            learning_rate)

new_weights_2 = weight_update_rule(multilayer_perceptron[2].weight,
                            multilayer_perceptron[2].weight.grad,
                            learning_rate)

# We reassign the weights to our model
multilayer_perceptron[0].weight = new_weights_0
multilayer_perceptron[2].weight = new_weights_2


# In[22]:


# Clears the previous gradients
multilayer_perceptron.zero_grad()


# In[23]:


# Let's produce a new prediction
new_y_hat = multilayer_perceptron(image_tensor)
new_y_hat.item()


# In[24]:


# Compare with the old
new_y_hat.item(), y_hat.item()


# In[25]:


loss_2 = absolute_error(y, new_y_hat)
loss_2


# In[26]:


loss_2, loss_1


# In[27]:


# We see that the loss has improved from the first attempt
loss_2 < loss_1


# ## Training Loop
# The training loop is the core process where the neural network learns from the data. It involves iteratively updating the model's weights based on the calculated loss. In this example, we manually implement the weight update rule and track the loss over epochs.

# In[28]:


from tqdm import tqdm

# Reinitialize the network
torch.manual_seed(42)
for layer in multilayer_perceptron:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)

# We repeat the process with a smaller learning rate
learning_rate = 1e-3
losses = []
best_loss = torch.inf
best_prediction = torch.tensor(-1)
for i in tqdm(range(200)):
    y_hat = multilayer_perceptron(image_tensor)
    loss = absolute_error(y, y_hat)
    if loss < best_loss:
        best_loss = loss
        best_prediction = y_hat
    loss.backward()
    new_weights_0 = weight_update_rule(multilayer_perceptron[0].weight,
                                multilayer_perceptron[0].weight.grad,
                                learning_rate)

    new_weights_2 = weight_update_rule(multilayer_perceptron[2].weight,
                                multilayer_perceptron[2].weight.grad,
                                learning_rate)

    # We reassign the weights to our model
    multilayer_perceptron[0].weight = new_weights_0
    multilayer_perceptron[2].weight = new_weights_2

    # We remove the gradients that we computed in the previous update
    multilayer_perceptron.zero_grad()

    losses.append(loss.item())


# In[29]:


# Print the best prediction
error = absolute_error(y, best_prediction)
print("# Manual update")
print("Target", y.item())
print("Best Prediction", best_prediction.item())
print("Error", error.item())


# In[30]:


# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o', linestyle='-', markersize=4)
plt.title('Training Loss Over Epochs (Manual update)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# ## Stochastic Gradient Descent
# The manual weight update rule we implemented earlier is the core of the Stochastic Gradient Descent (SGD) optimizer. However, instead of manually updating weights, we can leverage the efficient [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) implementation provided by PyTorch. This built-in optimizer runs faster than our custom implementation.

# In[31]:


from torch.optim import SGD

# Reinitialize the network
torch.manual_seed(42)
for layer in multilayer_perceptron:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)

# We repeat the process with the SGD optimizer
# It uses the same logic as our manual update, but is implemented more efficiently
learning_rate = 1e-3
optimizer = SGD(multilayer_perceptron.parameters(), lr=learning_rate)
losses = []
best_loss = torch.inf
best_prediction = torch.tensor(-1)
for i in tqdm(range(200)):
    y_hat = multilayer_perceptron(image_tensor)
    loss = absolute_error(y, y_hat)
    if loss < best_loss:
        best_loss = loss
        best_prediction = y_hat
    loss.backward()

    # We reassign the weights to our model
    optimizer.step()

    # We remove the gradients that we computed in the previous update
    optimizer.zero_grad()

    losses.append(loss.item())


# In[32]:


# Print the best prediction
error = absolute_error(y, best_prediction)
print("# SGD update")
print("Target", y.item())
print("Best Prediction", best_prediction.item())
print("Error", error.item())


# In[33]:


# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o', linestyle='-', markersize=4)
plt.title('Training Loss Over Epochs (SGD update)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# ## Adam (Adaptive Gradients with Momentum)
# The Adam optimizer is a variant of stochastic gradient descent that often converges faster. We can leverage the [torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) implementation to benefit from its efficiency and adaptive learning rate adjustments.
# 
# To learn more about the details of Adam, it's good to refer to the [original paper](https://arxiv.org/abs/1412.6980) and watch [this video](https://www.youtube.com/watch?v=MD2fYip6QsQ).

# In[34]:


from torch.optim import Adam

# Reinitialize the network
torch.manual_seed(42)
for layer in multilayer_perceptron:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight)

# We repeat the process with a momentum-based optimizer (Adam)
learning_rate = 1e-3
optimizer = Adam(multilayer_perceptron.parameters(), lr=learning_rate)
losses = []
best_loss = torch.inf
best_prediction = torch.tensor(-1)
for i in tqdm(range(200)):
    y_hat = multilayer_perceptron(image_tensor)
    loss = absolute_error(y, y_hat)
    if loss < best_loss:
        best_loss = loss
        best_prediction = y_hat
    loss.backward()

    # We reassign the weights to our model
    optimizer.step()

    # We remove the gradients that we computed in the previous update
    optimizer.zero_grad()

    losses.append(loss.item())


# In[35]:


# Print the best prediction
error = absolute_error(y, best_prediction)
print("# Adam update")
print("Target", y.item())
print("Best Prediction", best_prediction.item())
print("Error", error.item())


# In[36]:


# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o', linestyle='-', markersize=4)
plt.title('Training Loss Over Epochs (Adam update)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


# ## Exploration
# 
# * Try training the model with higher and lower learning rates. This is commonly the most sensitive and important hyperparameter during neural network training.

# ## Acknowledgements
# 
# Many thanks to Johanna Reiml for her help discussing and improving this notebook.
