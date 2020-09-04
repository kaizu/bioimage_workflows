#!/usr/bin/env python
# coding: utf-8

# # Test spot detection (Generation)
# 
# This is a test suite.

# !pip uninstall -y scopyon
# !pip install git+https://github.com/ecell/scopyon
# !pip freeze

# In[1]:


import scopyon


# Set physical parameters.

# In[2]:


config = scopyon.DefaultConfiguration()
config.update("""
default:
    magnification: 360
    detector:
        exposure_time: 0.033
""")


# In[3]:


pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5


# Set the number of processes to enable `multiprocessing`:

# In[4]:


config.environ.processes = 20


# Prepare for generating inputs.

# In[5]:


import numpy
rng = numpy.random.RandomState(123)
N = 1000


# Collect data.

# In[6]:


from pathlib import Path
artifacts = Path("./artifacts")
artifacts.mkdir(parents=True, exist_ok=True)


# In[7]:


with open(artifacts / 'config.yaml', 'w') as f:
    f.write(repr(config))


# In[8]:


for i in range(10):
    inputs = rng.uniform(-L_2, +L_2, size=(N, 2))
    numpy.save(artifacts / f'inputs{i:03d}.npy', inputs)
    img, infodict = scopyon.form_image(inputs, config=config, rng=rng, full_output=True)
    img.save(str(artifacts / f'image{i:03d}.npy'))
    data = numpy.array([(row[2], row[3]) for row in infodict['true_data'].values()])
    numpy.save(artifacts / f'data{i:03d}.npy', data)

