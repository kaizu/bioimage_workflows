#!/usr/bin/env python
# coding: utf-8

# In[1]:


num_samples = 3
interval = 33.0e-3
seed = 123


# In[2]:


import pathlib
inputpath = pathlib.Path("./artifacts")
artifacts = pathlib.Path("./artifacts")
artifacts.mkdir(parents=True, exist_ok=True)


# In[3]:


import scopyon
config = scopyon.Configuration(filename=inputpath / "config.yaml")
pixel_length = config.default.detector.pixel_length / config.default.magnification


# In[4]:


import numpy
rng = numpy.random.RandomState(seed)


# In[5]:


def trace_spots(spots, ndim, rng):
    observation_vec = []
    lengths = []
    for i in range(len(spots[0])):
        iprev = i
        for j in range(1, len(spots)):
            displacements = numpy.power(spots[j][:, : ndim] - spots[j - 1][iprev, : ndim], 2).sum(axis=1)
            inext = displacements.argmin()
            displacement = numpy.sqrt(displacements[inext])
            intensity = rng.normal(1.0, 0.5)
            observation_vec.append([displacement, intensity])
            iprev = inext
        lengths.append(len(spots) - 1)
    return observation_vec, lengths


# In[6]:


observation_vec = []
lengths = []
ndim = 2
for i in range(num_samples):
    spots_ = numpy.load(inputpath / f"spots{i:03d}.npy")
    t = spots_[0, 0]
    spots = [[spots_[0, 1: ]]]
    for row in spots_[1: ]:
        if row[0] == t:
            spots[-1].append(row[1: ])
        else:
            t = row[0]
            spots[-1] = numpy.asarray(spots[-1])
            spots.append([row[1: ]])
    else:
        spots[-1] = numpy.asarray(spots[-1])
    # print(spots)

    observation_vec_, lengths_ = trace_spots(spots, ndim, rng)
    observation_vec.extend(observation_vec_)
    lengths.extend(lengths_)
observation_vec = numpy.array(observation_vec)


# In[7]:


print(len(lengths), sum(lengths))


# In[8]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[9]:


fig = make_subplots(rows=1, cols=2, subplot_titles=['Square Displacement', 'Intensity'])

fig.add_trace(go.Histogram(x=observation_vec[:, 0], nbinsx=100), row=1, col=1)

fig.add_trace(go.Histogram(x=observation_vec[:, 1], nbinsx=100), row=1, col=2)

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75, showlegend=False)
fig.show()


# In[ ]:





# In[10]:


from scopyon.analysis import PTHMM


# In[11]:


model = PTHMM(n_diffusivities=4, n_oligomers=1, n_iter=100, random_state=rng)
model.fit(observation_vec, lengths)


# In[12]:


print("diffusivities=\n", model.diffusivities_)
print("D=\n", pixel_length ** 2 * model.diffusivities_ / interval / 1e-12)


# In[13]:


print("intensity_means=", model.intensity_means_)
print("intensity_vars=", model.intensity_vars_)


# In[14]:


print("startprob=\n", model.startprob_)


# In[15]:


P = model.transmat_
k = -numpy.log(1 - P) / interval
k.ravel()[:: k.shape[0] + 1] = 0.0
print("transmat=\n", model.transmat_)
print("state_transition_matrix=\n", k)


# In[16]:


expected_vec = numpy.zeros((sum(lengths), 2), dtype=observation_vec.dtype)
for i in range(len(lengths)):
    X_, Z_ = model.sample(lengths[i])
    expected_vec[sum(lengths[: i]): sum(lengths[: i + 1])] = X_


# In[17]:


fig = make_subplots(rows=1, cols=2, subplot_titles=['Square Displacement', 'Intensity'])

fig.add_trace(go.Histogram(x=observation_vec[:, 0], nbinsx=100), row=1, col=1)
fig.add_trace(go.Histogram(x=expected_vec[:, 0], nbinsx=100), row=1, col=1)

fig.add_trace(go.Histogram(x=observation_vec[:, 1], nbinsx=100), row=1, col=2)
fig.add_trace(go.Histogram(x=expected_vec[:, 1], nbinsx=100), row=1, col=2)

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75, showlegend=False)
fig.show()

