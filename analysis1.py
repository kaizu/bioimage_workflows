#!/usr/bin/env python
# coding: utf-8

# In[1]:


num_samples = 3
interval = 33.0e-3
num_frames = 100


# In[2]:


nproc = 20


# In[3]:


import numpy
timepoints = numpy.linspace(0, interval * num_frames, num_frames + 1)


# In[4]:


import pathlib
inputpath = pathlib.Path("./artifacts")
artifacts = pathlib.Path("./artifacts")
artifacts.mkdir(parents=True, exist_ok=True)


# In[5]:


import scopyon


# In[6]:


import warnings
warnings.simplefilter('ignore', RuntimeWarning)

for i in range(num_samples):
    imgs = [scopyon.Image(data) for data in numpy.load(inputpath / f"images{i:03d}.npy")]
    spots = [
        scopyon.analysis.spot_detection(
            img.as_array(), processes=nproc,
            min_sigma=1, max_sigma=4, threshold=50.0, overlap=0.5)
        for img in imgs]

    spots_ = []
    for t, data in zip(timepoints, spots):
        spots_.extend(([t] + list(row) for row in data))
    spots_ = numpy.array(spots_)
    numpy.save(artifacts / f"spots{i:03d}.npy", spots_)
    
    print("{} spots are detected in {} frames.".format(len(spots_), len(imgs)))

warnings.resetwarnings()


# In[7]:


get_ipython().system('ls ./artifacts')

