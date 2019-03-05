"""
pt4_pclimate_change_impacts.py
--------------------------------------------------------------------------------
WORKSHOP PART 4: USING RANDOM FORESTS TO PREDICT CLIMATE CHANGE IMPACTS ON
POTENTIAL BIOMASS
This is the final interactive component of the Forests2020 workshop: "Mapping
restoration potential using Machine Learning". This workshop is based on the
open source programming language python, and utilises the geospatial library
xarray (http://xarray.pydata.org/en/stable/) and the machine learning library
scikit-learn (https://scikit-learn.org/stable/index.html).

In this component, we use a fitted random forest regression model to make a
prediction of the potential AGB that could be supported across a region should
natural ecosystems be allowed to regenerate fully. We then use the same model
to make a prediction as to how this might change under different climate change
scenarios. For intact old growth forest this is an estimate of the future C
sink. Note that the caveat here is that it does not account for any CO2
fertilisation effects.

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

import data_io as io
import set_training_areas as training_areas
import map_figures as mfig

"""
#===============================================================================
PART A: FIT POTENTIAL BIOMASS MODEL
This is a bit of a repeat from before again, but useful as a reminder.
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

"""
#===============================================================================
PART B: FIT CLIMATE CHANGE SCENARIO
Now load in new set of predictors, this time using the climate change scenarios
from WorldClim, rather than the regular climatologymodels we have been using
#-------------------------------------------------------------------------------
"""
scenario_name = 'rcp85' # choices are rcp26, rcp45, rcp60, rcp85
predictors_scenario,AGB,scenariomask,labels= io.load_predictors_scenarios(scenario_name)
# predict using rf model
AGBpot_scenario = rf.predict(predictors_scenario)

"""
#===============================================================================
PART C: PLOT POTENTIAL BIOMASS UNDER FUTURE CLIMATE SCENARIO
The scenarios represent downscaled climatology forecasts from the CMIP5
compilation for four different RCPs
(https://en.wikipedia.org/wiki/Representative_Concentration_Pathways),
available here: http://www.worldclim.org/cmip5_30s

We will use 2070 forecasts.
ref: Hijmans et al, 2005. Very high resolution interpolated climate surfaces
for global land areas. International Journal of Climatology 25: 1965-1978.

Note that for simplicity, we will only use the forecast downsampled from the
HADGEM2-ES model, not the full suite of CMIP5 simulations. Thus our potential
AGB results represent extrapolations based on one model simulation of future
climate, and should not be considered an accurate or robust reconstruction of
potential AGB. This is simpley to provide an example of the potential
applications of this methodology for looking at future AGB stocks.

For a more thorough analysis, check out this article by J-.F. Exbrayat in
Nature Scientific Reports: https://www.nature.com/articles/s41598-017-15788-6
#-------------------------------------------------------------------------------
"""
# As before, we'll load in an existing dataset to get the georeferencing information
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

#now do the same for the scenario
agbpot_scenario = agb.copy()
agbpot_scenario.values = np.zeros(landmask.shape)*np.nan
agbpot_scenario.values[scenariomask] = AGBpot_scenario.copy()

#now do the same for the potential difference
agbpot_difference = agb.copy()
agbpot_difference.values = agbpot_scenario.values - agbpot.values

# Then we plot up both maps for comparison
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
agbpot_scenario.plot(ax=axes[0], vmin=0, vmax=400, cmap='viridis', add_colorbar=True,
                    extend='max', cbar_kwargs={'label': 'AGB$_{pot}$  / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
agbpot_difference.plot(ax=axes[1], vmin=-100, vmax=100, cmap='bwr_r', add_colorbar=True,
                    extend='both', cbar_kwargs={'label': 'difference in AGB$_{pot}$ / Mg ha$^{-1}$',
                    'orientation':'horizontal'})
for ax in axes:
    ax.set_aspect("equal")
axes[0].set_title("Potential AGB in 2070\nunder %s" % scenario_name)
axes[1].set_title("Difference in potential AGB\nunder %s" % scenario_name)
plt.show()
