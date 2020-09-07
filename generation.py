#!/usr/bin/env python
# coding: utf-8

# !pip uninstall -y scopyon
# !pip install git+https://github.com/ecell/scopyon
# !pip freeze | grep scopyon

# In[1]:


seed = 123
num_samples = 3
exposure_time = 33.0e-3
interval = 33.0e-3
num_frames = 100
Nm = [100, 100, 100]
Dm = [0.222e-12, 0.032e-12, 0.008e-12]
transmat = [
    [0.0, 0.5, 0.0],
    [0.5, 0.0, 0.2],
    [0.0, 1.0, 0.0]]


# In[2]:


nproc = 20


# In[3]:


# !pip install mlflow


# In[4]:


import numpy
rng = numpy.random.RandomState(seed)


# In[5]:


import scopyon


# In[6]:


config = scopyon.DefaultConfiguration()
config.default.detector.exposure_time = exposure_time
pixel_length = config.default.detector.pixel_length / config.default.magnification
L_2 = config.default.detector.image_size[0] * pixel_length * 0.5
L_2


# In[7]:


config.environ.processes = nproc


# In[8]:


timepoints = numpy.linspace(0, interval * num_frames, num_frames + 1)
ndim = 2


# In[9]:


import pathlib
artifacts = pathlib.Path("./artifacts")
artifacts.mkdir(parents=True, exist_ok=True)


# In[10]:


config.save(artifacts / 'config.yaml')


# In[11]:


for i in range(num_samples):
    ret = scopyon.sample(timepoints, N=Nm, lower=-L_2, upper=+L_2, ndim=ndim, D=Dm, transmat=transmat, rng=rng)
    inputs = [(t, numpy.hstack((points[:, : ndim], points[:, [ndim + 1]], numpy.ones((points.shape[0], 1), dtype=numpy.float64)))) for t, points in zip(timepoints, ret)]
    imgs = list(scopyon.generate_images(inputs, num_frames=num_frames, config=config, rng=rng))
    
    inputs_ = []
    for t, data in inputs:
        inputs_.extend(([t] + list(row) for row in data))
    inputs_ = numpy.array(inputs_)
    numpy.save(artifacts / f"inputs{i:03d}.npy", inputs_)

    numpy.save(artifacts / f"images{i:03d}.npy", numpy.array([img.as_array() for img in imgs]))


# In[12]:


get_ipython().system('ls ./artifacts')

