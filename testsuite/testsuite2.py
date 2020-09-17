#!/usr/bin/env python
# coding: utf-8

# # Test diffusion
# 
# This is an example using `scopyon` (https://scopyon.readthedocs.io/).

# In[1]:


import scopyon


# Set the configuration first.

# In[2]:


config = scopyon.DefaultConfiguration()
config.default.magnification = 360.0
config.default.detector.exposure_time = 33.0e-3  # second


# In[3]:


# config.environ.processes = 20


# A field of microscopic view could be calculated as follows:

# In[4]:


pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5


# Randomly generate positions of 1000 molecules on membrane. The diffusion rate `D` is `0.1e-12`.

# In[5]:


import numpy
rng = numpy.random.RandomState(1)
num_frames = 50
duration = config.default.detector.exposure_time * 5
t = numpy.linspace(0, duration, num_frames + 1)


# In[6]:


N = 1000
D = 0.1e-12  # m ** 2 / s
inputs = scopyon.sample_inputs(t, N=N, lower=-L_2, upper=+L_2, ndim=2, D=D, rng=rng)


# `scopyon.form_image` generates a single image from the given inputs.

# In[7]:


# import logging
# logging.basicConfig(level=logging.INFO)


# In[8]:


img1 = scopyon.form_image(inputs, config=config, rng=rng)


# In[9]:


spots1 = scopyon.analysis.spot_detection(
    img1.as_array(), min_sigma=1, max_sigma=4, threshold=40.0, overlap=0.5)


# In[10]:


r = 6
shapes = [dict(x=spot[0], y=spot[1], sigma=r, color='red')
        for spot in spots1]
img1.show(shapes=shapes)


# Take another image at time 't=0.1'.

# In[11]:


img2 = scopyon.form_image(inputs, start_time=100e-3, config=config, rng=rng)


# In[12]:


spots2 = scopyon.analysis.spot_detection(
    img2.as_array(), min_sigma=1, max_sigma=4, threshold=40.0, overlap=0.5)


# Compare two images and choose the closest point for each points.

# In[13]:


closest = []
for spot in spots1:
    distance = spots2[:, : 2] - spot[: 2]
    idx = (distance ** 2).sum(axis=1).argmin()
    closest.extend(distance[idx])
closest = numpy.array(closest)


# In[14]:


closest


# A diffusion rate is estimated based on the average of closest distances (displacements). The units below is a square of pixels per second.

# In[15]:


D_ = numpy.average(closest ** 2) / (2 * duration)
D_


# Rescaled into physical units (square meters per second).

# In[16]:


D_ * (pixel_length ** 2)


# Plot data with probability densities.

# In[17]:


import plotly.graph_objects as go
from scipy.stats import norm

fig = go.Figure()

fig.add_trace(go.Histogram(x=closest, histnorm='probability density', name='Samples'))

x = numpy.linspace(closest.min(), closest.max(), 101)
y = norm.pdf(x, scale=numpy.sqrt(2 * D_ * duration))
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Estimated'))

y = norm.pdf(x, scale=numpy.sqrt(2 * (D / (pixel_length ** 2)) * duration))
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='True'))

fig.show()

