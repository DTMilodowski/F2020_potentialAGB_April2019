"""
pt3_potential_agb_maps.py
--------------------------------------------------------------------------------
WORKSHOP PART 3: PREDICTING POTENTIAL BIOMASS AND RESOTRATION POTENTIAL
This is the third interactive component of the Forests2020 workshop: "Mapping
restoration potential using Machine Learning". This workshop is based on the
open source programming language python, and utilises the geospatial library
xarray (http://xarray.pydata.org/en/stable/) and the machine learning library
scikit-learn (https://scikit-learn.org/stable/index.html).

In this component, we use a fitted random forest regression model to make a
prediction of the potential AGB that could be supported across a region should
natural ecosystems be allowed to regenerate fully. We then calculate the
deficit, which is the difference between the observed and potential AGB and
represents a data-driven estimate of the restoration potential with respect to
C sequestration.

Finally, we'll finish up by saving the raster layers to geotiffs that can be
loaded up onto an EO lab.

The detailed code underlying much of the following analyses are stashed away in
the source code, so you don't need to worry about them too much. If you are
interested, you are welcome to browse this at your leisure.

05/03/2019 - D. T. Milodowski
--------------------------------------------------------------------------------
"""

"""
# Import the necessary packages
"""
import numpy as np                  # standard package for scientific computing
import xarray as xr                 # xarray geospatial package
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package
#sns.set()                           # set some nice default plotting options

# Import some parts of the scikit-learn library
from sklearn.ensemble import RandomForestRegressor

# Import custom libaries
import sys
sys.path.append('./random_forest/')
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')
sys.path.append('./eolab/')

import data_io as io
import set_training_areas as training_areas
import map_figures as mfig
import prepare_EOlab_layers as eo

"""
#===============================================================================
PART A: POTENTIAL BIOMASS ESTIMATION
This is a bit of a repeat from before, except we'll fit the model using all the
data available, rather than splitting into cal-val sets.
Fits random forest regression models to estimate potential biomass stocks.
#-------------------------------------------------------------------------------
"""
# First of all, let's load in some data again
predictors,AGB,landmask,labels=io.load_predictors()

# Now create the training set
# First create training mask based on Hinterland Forest Landscapes mapped by
# Tyukavina et al (2015, Global Ecology and Biogeography) and stable non-forest
# (natural) classes from ESA CCI land cover dataset
training_mask= training_areas.set()

# Now subset the predictors and AGB data according to this training mask
X = predictors[training_mask[landmask]]
y = AGB[training_mask[landmask]]

# create the random forest object with predefined parameters
# *** = parameters that often come out as being particularly sensitive
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth=None,            # ***maximum number of branching levels within each tree
            max_features='auto',       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=0.0, # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=5,       # ***The minimum number of samples required to be at a leaf node
            min_samples_split=20,       # ***The minimum number of samples required to split an internal node
            min_weight_fraction_leaf=0.0,
            n_estimators=100,          # ***Number of trees in the random forest
            n_jobs=-1,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=None,         # seed used by the random number generator
            verbose=0)

# fit the model
rf.fit(X,y)
cal_score = rf.score(X,y) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R$^2$ = %.02f" % cal_score)

# Now the model has been fitted, we will predict the potential AGB across the
# full dataset
AGBpot = rf.predict(predictors)

# Now lets plot this onto a map
# We'll load in an existing dataset to get the georeferencing information
agb_file = '../data/agb/colombia_Avitabile_AGB_2km.tif' # the agb file

# open file and store data in an xarray called agb
agb_ds = xr.open_rasterio(agb_file)
agb = agb_ds.sel(band=1)

# rename coordinates to latitude and longitude
agb = agb.rename(x='longitude',y='latitude')
agb.values[agb.values<0]=np.nan

#let's copy to a new xarray for AGBpot
agbpot = agb.copy()
agbpot.values = np.zeros(landmask.shape)*np.nan
agbpot.values[landmask] = AGBpot.copy()

# Then we plot up both maps for comparison
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
agb.plot(ax=axes[0], vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'AGB / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
agbpot.plot(ax=axes[1], vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'AGB$_{pot}$ / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
for ax in axes:
    ax.set_aspect("equal")
axes[0].set_title("Observed AGB")
axes[1].set_title("Modelled potential AGB")
plt.show()

"""
#===============================================================================
PART B: CALCULATING THE DEFICIT AND RESTORATION POTENTIAL
The deficit is simply the difference between AGB and AGBpot. It is
straightforward to add and subtract xarrays
#-------------------------------------------------------------------------------
"""
#let's copy to a new xarray for AGBdef (the deficit)
agbdef = agb.copy()
agbdef.values = agbpot.values-agb.values

fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
agbdef.plot(ax=axis, vmin=-200, vmax=200, cmap='bwr_r', add_colorbar=True,
                    extend='both', cbar_kwargs={'label': 'AGB$_{def}$ / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
axis.set_aspect("equal")
axis.set_title("AGB deficit")
plt.show()


"""
#===============================================================================
PART C: PRODUCING EO LABORATORY LAYERS
Save layers as geotiffs suitable for uploading onto EO lab apps.
#-------------------------------------------------------------------------------
"""
output_prefix = '../EOlab_layers/F2020_workshop_AGBpot' # a prefix for your output file.
cmap = 'viridis' # the colormap you want to use
ulim=400 # the upper limit of the colormap
llim=0 # the lower limit of the colormap
colorbar_label = 'AGB$_{potential}$ / Mg ha$^{-1}$' # the lael for the colorbar
# Note that writing a display layer automatically creates the corresponding
# data layer
eo.write_array_to_display_layer_GeoTiff(agbpot,output_prefix,cmap,ulim,llim)
eo.plot_legend(cmap,ulim,llim,colorbar_label, output_prefix,extend='max')
