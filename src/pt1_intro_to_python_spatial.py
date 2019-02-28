"""
pt1_intro_to_python_spatial.py
--------------------------------------------------------------------------------
WORKSHOP PART 1: INTRODUCTION TO SPATIAL ANALYSIS IN PYTHON
This is the first interactive component of the Forests2020 workshop: "Mapping
restoration potential using Machine Learning". This workshop is based on the
open source programming language python, and utilises the geospatial library
xarray (http://xarray.pydata.org/en/stable/) and the machine learning library
scikit-learn (https://scikit-learn.org/stable/index.html).

28/02/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
NOTE!!!
This is a comment block, bookended by three double quotation marks
Comment blocks and comments are ignored by python, but are useful for explaining
what the code is doing
"""

# this is a comment, with a hash at the start. The line after the hash is
# ignored

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
sns.set()                           # set some nice default plotting options

"""
Part 1A: Loading a dataset using xarray
We are going to start by loading a raster dataset into python using xarray,
before exploring how to interact with this xarray object. For later scripts, a
lot of the processing will be tucked away inside other functions, but it is
useful to know what we are dealing with.
"""

# To open a raster dataset is easy enough to do. We need the name of the file in
# question, alongside it's full path
path2root = '../' # this is the path to the root directory of the tutorial
                  # Note that '../' points to one directory level above the
                  # current working directory (i.e. a 'relative' path)
path2data = path2root + 'data/' # strings can be added together
print(path2data) # print path2data to screen

agb_file = path2data + 'agb/colombia_Avitabile_AGB_2km.tif' # the agb file
print(agb_file) # print filename to screen

# open file and store data in an xarray called agb
agb = xr.open_rasterio(agb_file)
print(type(agb))

# Let's explore the xarray structure a little
# The key properties for a data array are:
# 1) the values numpy array containing the gridded observations
print(type(agb.values))
# 2) the dimensions of the array
print(agb.dims)
# 3) the coordinates of the data
print(agb.coords)
# 4) the meta data e.g. coordinate system
print(agb.attrs)

# We'll explore xarray interactions further in due course, but for now, it is
# worthwhile simply plotting a map of the data. With xarray, this is really easy.

"""
Basic plotting with xarray
"""

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(14,12))



agb_mask = agb.values[0]!=np.float32(agb.nodatavals[0])
