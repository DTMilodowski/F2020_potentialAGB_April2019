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
ds = xr.open_rasterio(agb_file)
print(type(ds))

# Let's explore the xarray structure a little
# The key properties for a data array are:
# 1) the values numpy array containing the gridded observations
print(type(ds.values))
# 2) the dimensions of the array
print(ds.dims)
# 3) the coordinates of the data
print(ds.coords)
# 4) the meta data e.g. coordinate system
print(ds.attrs)
# 5) the nodata value
print(ds.nodatavals)

# There was only one band in the geotiff, with the AGB values. We can select
# this band very easily using the sel() function
agb = ds.sel(band=1)

# rename coordinates to latitude and longitude
agb = agb.rename(x='longitude',y='latitude')

# convert nodatavalues to numpy-recognised nodata (np.nan)
agb.values[agb.values<0]=np.nan

# This new xarray now only has the dimensions y and x (longtitude and latitude)
print(ds.coords)
print(agb.coords)

# We'll explore xarray interactions further in due course, but for now, it is
# worthwhile simply plotting a map of the data. With xarray, this is really easy.

"""
Basic plotting with xarray
"""

fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
agb.plot(ax=axis, vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'AGB / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
axis.set_aspect("equal")
plt.show()

"""
Subsets of xarrays
"""
# OK, now we can try to manipulate this a bit. It is quite easy to select a
# subset of an xarray - see http://xarray.pydata.org/en/stable/indexing.html
# We are not going to go through every example here, but a simple spatial
# subset can be taken if we know the coordinate bounds
min_lat = 0
max_lat = 7
min_long = -77
max_long = -60
agb_subset = agb.sel(latitude=slice(max_lat,min_lat),longitude = slice(min_long,max_long))
# Note that since our latitudes are listed in decreasing order, we take the
# slice from max to min, which might seem initially counter-intuitive.
fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
agb_subset.plot(ax=axis, vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'AGB / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
axis.set_aspect("equal")
plt.show()

# of course, we could have achieved the same plot without creating a new object:
fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
agb.sel(latitude=slice(max_lat,min_lat),longitude = slice(min_long,max_long)).plot(
                    ax=axis, vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'AGB / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
axis.set_aspect("equal")
plt.show()

"""
Now try plotting some of the other datasets that we are going to use.
We have the following data that will be used in the workshop. You don't need to
plot them all.

WorldClim2 climatology (http://www.worldclim.org/bioclim):
    filename = '../data/climatology/colombia_wc2.0_bio_30s_??_2km.tif'
        where ?? refers to a 2 digit number between 01 and 19, indicating the
        corresponding characteristic
            BIO1 = Annual Mean Temperature
            BIO2 = Mean Diurnal Range (Mean of monthly (max temp - min temp))
            BIO3 = Isothermality (BIO2/BIO7) (* 100)
            BIO4 = Temperature Seasonality (standard deviation *100)
            BIO5 = Max Temperature of Warmest Month
            BIO6 = Min Temperature of Coldest Month
            BIO7 = Temperature Annual Range (BIO5-BIO6)
            BIO8 = Mean Temperature of Wettest Quarter
            BIO9 = Mean Temperature of Driest Quarter
            BIO10 = Mean Temperature of Warmest Quarter
            BIO11 = Mean Temperature of Coldest Quarter
            BIO12 = Annual Precipitation
            BIO13 = Precipitation of Wettest Month
            BIO14 = Precipitation of Driest Month
            BIO15 = Precipitation Seasonality (Coefficient of Variation)
            BIO16 = Precipitation of Wettest Quarter
            BIO17 = Precipitation of Driest Quarter
            BIO18 = Precipitation of Warmest Quarter
            BIO19 = Precipitation of Coldest Quarter

SoilGrids data (https://soilgrids.org/):
    filename = '../data/soils/'
    where ? refers to one of the following files:
        colombia_BDRICM_M_2km.tif
        colombia_BDRLOG_M_2km.tif
        colombia_BDTICM_M_2km.tif

        colombia_CLYPPT_M_sl1_2km.tif # clay fractions at different depths
        colombia_CLYPPT_M_sl2_2km.tif
        colombia_CLYPPT_M_sl3_2km.tif
        colombia_CLYPPT_M_sl4_2km.tif
        colombia_CLYPPT_M_sl5_2km.tif
        colombia_CLYPPT_M_sl6_2km.tif
        colombia_CLYPPT_M_sl7_2km.tif

        colombia_SLTPPT_M_sl1_2km.tif # silt fractions at different depths
        colombia_SLTPPT_M_sl2_2km.tif
        colombia_SLTPPT_M_sl3_2km.tif
        colombia_SLTPPT_M_sl4_2km.tif
        colombia_SLTPPT_M_sl5_2km.tif
        colombia_SLTPPT_M_sl6_2km.tif
        colombia_SLTPPT_M_sl7_2km.tif

        colombia_SNDPPT_M_sl1_2km.tif # sand fractions at different depths
        colombia_SNDPPT_M_sl2_2km.tif
        colombia_SNDPPT_M_sl3_2km.tif
        colombia_SNDPPT_M_sl4_2km.tif
        colombia_SNDPPT_M_sl5_2km.tif
        colombia_SNDPPT_M_sl6_2km.tif
        colombia_SNDPPT_M_sl7_2km.tif

Hinterland forest extent (https://glad.umd.edu/dataset/hinterland-forests-2013)
    filename = '../data/landcover/colombia_HFL_2013_2km.tif'

ESACCI landcover (https://maps.elie.ucl.ac.be/CCI/viewer/)
    filename = '../data/landcover/colombia-ESACCI-LC-L4-LCCS-Map-P1Y-????-v2.0.7-2km-mode-lccs-class_2km.tif'
    where ??? refers to a year between 1992 and 2015

Note that the landcover data is coded by integer values uniqueto each of the
ESA-CCI landcover classes. Don't worry about this for now, as it is dealt with
elsewhere in the code. Forest classes are between XXX and XXX
"""
