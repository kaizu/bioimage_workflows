#!/usr/bin/env python
# coding: utf-8

# # Test spot detection (Evaluation)
# 
# This is a test suite.

# !pip uninstall -y scopyon
# !pip install git+https://github.com/ecell/scopyon
# !pip freeze | grep scopyon

# In[1]:


import numpy
import scopyon


# Set physical parameters.

# In[9]:


from pathlib import Path
inputpath = Path("./artifacts")

artifacts = Path("./artifacts")
artifacts.mkdir(parents=True, exist_ok=True)


# In[3]:


config = scopyon.Configuration(str(inputpath / 'config.yaml'))


# In[4]:


pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5


# Collect data.

# In[10]:


res = []
for i in range(10):
    img = scopyon.Image.load(str(inputpath / f'image{i:03d}.npy'))
    data = numpy.load(inputpath / f'data{i:03d}.npy')
    
    spots = scopyon.analysis.spot_detection(
        img.as_array(), min_sigma=1, max_sigma=4, threshold=40.0, overlap=0.5)
    numpy.save(artifacts / f'spots{i:03d}.npy', spots)
    res.append((img, data, spots))


# In[11]:


res


# Find closest true positions from spots detected.

# In[7]:


closest = []
for img, data, spots in res:
    for spot in spots:
        distance = data - spot[0: 2]
        idx = (distance ** 2).sum(axis=1).argmin()
        closest.append(distance[idx])
closest = numpy.array(closest).T


# Calculate average and std for the accuracy of the spot detection method.

# In[8]:


print(f"Average along x-axis = {numpy.average(closest[0]):+.5f} pixels, std = {numpy.std(closest[0])}")
print(f"Average along y-axis = {numpy.average(closest[1]):+.5f} pixels, std = {numpy.std(closest[1])}")


# Show the heat map.

# In[ ]:


import plotly.express as px
H, xedges, yedges = numpy.histogram2d(x=closest[0], y=closest[1], bins=41, range=[[-2, +2], [-2, +2]])
fig = px.imshow(H, x=(xedges[: -1]+xedges[1: ])*0.5, y=(yedges[: -1]+yedges[1: ])*0.5)
fig.show()

