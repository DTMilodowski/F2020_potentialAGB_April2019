"""
pt2_random_forest_basics.py
--------------------------------------------------------------------------------
WORKSHOP PART 2: FITTING AND INTERPRETTING RANDOM FOREST REGRESSION MODELS
This is the second interactive component of the Forests2020 workshop: "Mapping
restoration potential using Machine Learning". This workshop is based on the
open source programming language python, and utilises the geospatial library
xarray (http://xarray.pydata.org/en/stable/) and the machine learning library
scikit-learn (https://scikit-learn.org/stable/index.html).

There are two parts to this component. The first introduces the random forest
implementation in scikit-learn and fits some simple regression models so that
we can explore some of the limitations associated with the method and compare
it against the (hopefully) more familiar linear regression model.

The detailed code underlying much of the following analyses are stashed away in
the source code, so you don't need to worry about them too much. If you are
interested, you are welcome to browse this at your leisure.

28/02/2019 - D. T. Milodowski
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import eli5
from eli5.sklearn import PermutationImportance

# Import custom libaries

import sys
sys.path.append('./random_forest/')
sys.path.append('./data_io/')
sys.path.append('./data_visualisation/')

import data_io as io
import utility as util
import set_training_areas as training_areas
import general_plots as gplt
"""
import cal_val as cv                # a set of functions to help with calibration and validation
import random_forest as rf          # a set of functions to help fit and interpret random forest models
"""
"""
#===============================================================================
PART A: FITTING SIMPLE RANDOM FOREST REGRESSION MODELS & BASIC CAL-VAL
A toy example; comparing random forest regression and linear regression models
#-------------------------------------------------------------------------------
"""
X1,y1,X2,y2,X3,y3 = util.generate_simple_test_data()

# Let's just plot up these trial datsets so we can see what we are dealing with
fig1,axes=gplot.plot_test_data(X1,y1,X2,y2,X3,y3)

# Now lets fit a very simple random forest regression model to this data
X1=X1.reshape(-1, 1)
X2=X2.reshape(-1, 1)
X3=X3.reshape(-1, 1)
X_test = np.arange(0,10,0.1).reshape(-1, 1)

rf1 = RandomForestRegressor()
rf1.fit(X,y1)
rf2 = RandomForestRegressor()
rf2.fit(X,y2)
rf3 = RandomForestRegressor()
rf3.fit(X3,y3)
y1_test = rf1.predict(X_test)
y2_test = rf2.predict(X_test)
y3_test = rf3.predict(X_test)

# for comparison, plot linear regression for cases 1 and 3
lm1 = LinearRegression()
lm3 = LinearRegression()
lm1.fit(X,y1)
lm3.fit(X3,y3)
y1_test_lm = lm1.predict(X_test)
y3_test_lm = lm3.predict(X_test)

# and plot the results
fig2,axes = gplt.plot_test_data_with_regression_results(X1,y1,X2,y2,X3,y3,X_test,
                        y1_test,y1_test_lm,y2_test,y3_test,y3_test_lm,show=True):


# You should see that there are the following features
# 1) Able to fit complex non-linear functions, without specifying functional
#    relationship
# 2) Tendency to overfit with default hyperparameters - would want to conduct
#    a proper calibration-validation procedure to test which options to choose.
# 3) Inability to extrapolate outside of the parameter space occupied by the
#    training set
# 4) Inability to interpolate across large gaps in parameter space (random
#    forest algorithm tends to cluster data)
# 5) Tendency to under-predict extremes

# Let's run a quick calibration-validation procedure. Here we hold back a
# fraction of the data from the model fitting and use this to test the fitted
# random forest model on independent data. This allows an assessment of
# overfitting, which is characteristically indicated in a reduction in the
# quality of fit between the calibration and validation set.

# Scikit-learn has tools already coded up to help with this process.

# We will focus on the non-linear example (example 2 from above) in the first
# instance

# Split the training data into a calibration and validation set using the scikit learn toolbox
# in this case we will use a 50-50 split. In real applications one might
# consider using k-fold cross-validation (available in scikit-learn toolbox)
X_train, X_test, y_train, y_test = train_test_split(X, y2, train_size=0.5,
                                                    test_size=0.5)
#create the random forest object with predefined parameters
rf = RandomForestRegressor()
# fit the calibration sample
rf.fit(X_train,y_train)
y_train_rf = rf.predict(X_train)
cal_score = rf.score(X_train,y_train) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R$^2$ = %.02f" % cal_score)
y_test_rf = rf.predict(X_test)
val_score = rf.score(X_test,y_test)
print("Validation R$^2$ = %.02f" % val_score)
# Plot the calibration and validation data
# - First put observations and model values into dataframe for easy plotting with seaborn functions
calval_df = pd.DataFrame(data = {'val_obs': y_test,
                                 'val_model': y_test_rf,
                                 'cal_obs': y_train,
                                 'cal_model': y_train_rf})

fig3,axes= plt.subplots(nrows=1,ncols=2,figsize=[8,4])
sns.regplot(x='cal_obs',y='cal_model',data=calval_df,marker='+',
            truncate=True,ci=None,ax=axes[0])
axes[0].annotate('calibration R$^2$ = %.02f\nRMSE = %.02f' %
            (cal_score,np.sqrt(mean_squared_error(y_train,y_train_rf))),
            xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='left', verticalalignment='top')
sns.regplot(x='val_obs',y='val_model',data=calval_df,marker='+',
            truncate=True,ci=None,ax=axes[1])
axes[1].annotate('validation R$^2$ = %.02f\nRMSE = %.02f'
            % (val_score,np.sqrt(mean_squared_error(y_test,y_test_rf))),
            xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='left', verticalalignment='top')
axes[0].axis('equal')
axes[1].axis('equal')
fig3.tight_layout()
fig3.show()


# If you like, you can try to tweak the random forest hyperparameters to see if
# the overfiting can be reduced.
# The potential hyperparameters are indicated below. Those noted with a *** are
# particularly important. I'd suggest starting with n_estimators (the number of
# trees used to construct the random forest) and min_samples_split (the minimum
# number of data points within a leaf node before it gets split).
rf = RandomForestRegressor(bootstrap=True,
            criterion='mse',           # criteria used to choose split point at each node
            max_depth=None,            # ***maximum number of branching levels within each tree
            max_features='auto',       # ***the maximum number of variables used in a given tree
            max_leaf_nodes=None,       # the maximum number of leaf nodes per tree
            min_impurity_decrease=0.0, # the miminum drop in the impurity of the clusters to justify splitting further
            min_impurity_split=None,   # threshold impurity within an internal node before it will be split
            min_samples_leaf=20,        # ***The minimum number of samples required to be at a leaf node
            min_samples_split=2,       # ***The minimum number of samples required to split an internal node
            min_weight_fraction_leaf=0.0,
            n_estimators=100,          # ***Number of trees in the random forest
            n_jobs=-1,                 # The number of jobs to run in parallel for both fit and predict
            oob_score=True,            # use out-of-bag samples to estimate the R^2 on unseen data
            random_state=None,         # seed used by the random number generator
            verbose=0,
            wa_start=False)

# To do this properly would obviously be very time consuming and tedious.
# scikit-learn have programmed automatic routines to search through the
# hyperparameter options and find a near-optimal calibration (RandomizedSearch
# and GridSearch).
# These are beyond the scope of this workshop, but feel free to check them out
# if you want to do this yourself.

"""
#===============================================================================
PART B: POTENTIAL BIOMASS ESTIMATION
Fit random forest regression models to estimate potential biomass stocks.
Calibrate and validate potential biomass models
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

# Split the training data into a calibration and validation set using the scikit learn tool
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
#create the random forest object with predefined parameters
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
            verbose=0,
            wa_start=False)

# fit the calibration sample
rf.fit(X_train,y_train)
y_train_rf = rf.predict(X_train)
cal_score = rf.score(X_train,y_train) # calculate coefficeint of determination R^2 of the calibration
print("Calibration R$^2$ = %.02f" % cal_score)

# fit the validation sample
y_test_rf = rf.predict(X_test)
val_score = rf.score(X_test,y_test)
print("Validation R$^2$ = %.02f" % val_score)

# Plot the calibration and validation data
# - First put observations and model values into dataframe for easy plotting with seaborn functions
cal_df = pd.DataFrame(data = {'cal_obs': y_train,
                              'cal_model': y_train_rf})
val_df = pd.DataFrame(data = {'val_obs': y_test,
                              'val_model': y_test_rf})


fig4,axes= plt.subplots(nrows=1,ncols=2,figsize=[8,4])
sns.regplot(x='cal_obs',y='cal_model',data=cal_df,marker='.',
            truncate=True,ci=None,ax=axes[0],
            scatter_kws={'alpha':0.01,'edgecolor':'none'},
            line_kws={'color':'k'})
axes[0].annotate('calibration R$^2$ = %.02f\nRMSE = %.02f' %
            (cal_score,np.sqrt(mean_squared_error(y_train,y_train_rf))),
            xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='left', verticalalignment='top')
sns.regplot(x='val_obs',y='val_model',data=val_df,marker='.',
            truncate=True,ci=None,ax=axes[1],
            scatter_kws={'alpha':0.01,'edgecolor':'none'},
            line_kws={'color':'k'})
axes[1].annotate('validation R$^2$ = %.02f\nRMSE = %.02f'
            % (val_score,np.sqrt(mean_squared_error(y_test,y_test_rf))),
            xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='left', verticalalignment='top')
axes[0].axis('equal')
axes[1].axis('equal')
fig4.tight_layout()
fig4.show()

"""
#===============================================================================
# PART C: HOW IMPORTANT ARE DIFFFERENT VARIABLES?
#-------------------------------------------------------------------------------
# A really useful feature of random forest algorithms is that they allow one to
# estimate the relative importance of each of the explanatory variables. This
# can be useful both for understanding the system, and also for simplifying the
# model if desired.
# The default scikit-learn implemntation uses the mean decrease in impurity
# importance of each variable, computed by measuring how effective the variable
# is at reducing uncertainty the variance when creating the decision trees
#
# Instead of the gini importance, which can be biased, a more rigorous, but
# intensive approach is to use the permutation importance (Strobl et al (2007,
# BMC Bioinformatics); see also https://explained.ai/rf-importance/index.html).
# This estimates importance based on how the fit quality drops when a variable
# is unavailable.
#
# Note there are some caveats - if there are colinear variables, the importance
# values may not be so straightforward to interpret as this deflates the
# importance values, due to the variance being equally well described by
# multiple explanatory variables.
#
# Orthogonalisation of the predictors (e.g. running a PCA) may therefore be a
# useful preprocessing step.
#
# Also, if you have a weak model, the importances could vary substantially if
# you try to repeat the calibration
"""
# VARIABLE IMPORTANCES
perm = PermutationImportance(rf).fit(X_test, y_test)
# note to access all score decreases across all permutations, use perm.results_
# For now, we'll just deal with means and standard deviations
imp_df = pd.DataFrame(data = {'variable': labels,
                              'permutation_importance': perm.feature_importances_,
                              'gini_importance': rf.feature_importances_})

fig5,axes= plt.subplots(nrows=1,ncols=2,figsize=[8,8],sharex=True)
sns.barplot(x='permutation_importance',y='variable',data=imp_df,ax=axes[0])
axes[0].annotate('permutation importance',
            xy=(0.95,0.98), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='right', verticalalignment='top')
sns.barplot(x='gini_importance',y='variable',data=imp_df,ax=axes[1])
axes[1].annotate('gini importance',
            xy=(0.95,0.98), xycoords='axes fraction',backgroundcolor='none',
            horizontalalignment='right', verticalalignment='top')
plt.setp(axes[1].get_yticklabels(),visible=False)
axes[1].yaxis.set_ticks_position('left')
fig5.tight_layout()
fig5.show()

"""
#===============================================================================
# PART D: USING PARTIAL DEPENDENCIES TO UNDERSTAND FUNCTIONAL EFFECTS
#-------------------------------------------------------------------------------
# Importances tell you which variables were important in determining the target
# variable, but we still don't know how these explanatory variables effect the
# system (just that they are important). This is one drawback of using a non-
# parametric machine learning algorithm.
# One way to sort-of sidestep this difficulty is to use partial dependencies.
# Partial dependencies are the random forest equivalent to partial differentials
# of multivariate equations.
# We can plot partial dependencies by holding all variables but for the one of
# interest constant (e.g. the mean values for the other variables), then
# fitting the model across the range of the variable of interest. We'll try
# this now.
"""
# Variable in position 11 (WClim2_12) exhibited the strongest importance, so
# we'll try with this first
n_variables=X.shape[1]
var_ = np.linspace(np.min(X[:,11]),np.max(X[:,11])+1,200)
X_RM = np.zeros((var_.size,n_variables))
for i in range(0,n_variables):
    if i == 11:
        X_RM[:,i] = var_.copy()
    else:
        X_RM[:,i] = np.mean(X[:,i])

# predict with rf model
y_RM = rf.predict(X_RM)
# now plot
fig6,ax = plt.subplots(figsize=[8,5])
ax.plot(var_, y_RM,'-')
ax.set_xlabel(labels[11])
ax.set_ylabel("AGB / Mg ha$^{-1}$")
fig6.tight_layout()
fig6.show()

# Really we would want to plot a number of lines for different combinations of
# the other explanatory variables, rather than just the average, since the
# relationship may not hold across the entire parameter space (and could be
# quite different!)

# Lets do that quickly now
fig7,ax = plt.subplots(figsize=[8,5])

N_iterations = 20
for i in range(0,N_iterations):
    sample_row = np.random.randint(0,X.shape[0]+1)
    X_i = np.zeros((var_.size,n_variables))
    for j in range(0,n_variables):
        if j == 11:
            X_i[:,j] = _.copy()
        else:
            X_i[:,j] = (X[sample_row,j])

    # predict with rf model
    y_i = rf.predict(X_i)
    ax.plot(var_, y_i,'-',c='0.5',linewidth=0.5,alpha=0.8)

ax.plot(var_, y_RM,'-') # also plot line from before for comparison
ax.set_xlabel(labels[11])
ax.set_ylabel("AGB / Mg ha$^{-1}$")
fig7.tight_layout()
fig7.show()

# If you like, you could try this for another variable to see how they look
