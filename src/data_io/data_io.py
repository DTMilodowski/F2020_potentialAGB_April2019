"""
27/02/2019 - DTM
Paring back scripts to the data io routines required in the potential biomass
estimation.

30/11/2018 - DTM
Rewritten some of the functions specific to Forests2020 potential biomass work
- no LUH data
- restricted set of soilgrids parameters

12/11/2018 - JFE
This file contains the definition of some useful functions
for the pantrop-AGB-LUH work
"""

import xarray as xr #xarray to read all types of formats
from affine import Affine
import glob
import numpy as np
import sys
import set_training_areas
import rasterio
from copy import deepcopy

"""
load_predictors
This function loads all of the datasets containign the explanatory variables,
and applies a mask so that only land areas are considered, and any nodata values
are removed.

The function takes one input argument:
    path2root
This is the file path to the root directory for the workshop, and can be
expressed either as a full path or relative path. It is needed so that the code
knows exactly where things are stored - note the correct directory structure is
required!!!
By default, we assume that the working directory is the "src" directory, in
which case the path2root is simply "../"

The function returns two objects:
    1) predictors: a large 2D numpy array where the rows correspond with land
       pixels and the columns correspond with each successive data set.
    2) agb: a large 1D numpy array with the AGB estimate for each land pixel.
       Note that the pixel order matches the pixel order in the predictors
       array.
    3) landmask: a boolian array which dimensions (n_latitude,n_longitude)
       where land pixels are marked by ones, and water bodies/nodata pixels
       are marked by zeros
"""
def load_predictors(path2root = "../"):

    # Path structures
    path2wc = path2root+'/data/climatology/'
    path2sg = path2root+'/data/soils/'
    path2agb = path2root+'data/agb/'

    # Load the worldclim2 data
    nodata=[]
    for ff in sorted(glob.glob(path2wc+'*tif')):
        nodata.append(rasterio.open(ff).nodatavals[0])

    wc2 = xr.concat([xr.open_rasterio(f) for f in sorted(glob.glob(path2wc+'*tif'))],dim='band')
    wc2_mask = wc2[0]!=nodata[0]
    for ii in range(wc2.shape[0]):
        wc2_mask = wc2_mask & (wc2[ii]!=nodata[ii])
    print('Loaded WC2 data')

    # Load the soilgrids data
    # Note that we filter out a bunch of variables correlated with land cover
    soilfiles_all = glob.glob(path2sg+'*tif')
    soilfiles = []
    #             %sand %silt %clay %D2Rhorizon %probRhorizon %D2bedrock
    filtervars = ['SNDPPT','SLTPPT','CLYPPT','BDRICM','BDRLOG','BDTICM']
    for ff in range(len(soilfiles_all)):
        if soilfiles_all[ff].split('/')[-1].split('.')[0].split('_')[0] in filtervars:
            soilfiles.append(soilfiles_all[ff])

    nodata=[]
    for ff in sorted(soilfiles):
        nodata.append(rasterio.open(ff).nodatavals[0])

    soil= xr.concat([xr.open_rasterio(f) for f in sorted(soilfiles)],dim='band')
    soil_mask = soil[0]!=nodata[0]
    for ii in range(soil.shape[0]):
        soil_mask = soil_mask & soil[ii]!=nodata[0]
    print('Loaded SOILGRIDS data')

    #also load the AGB data to check we only keep pixels with AGB estimates
    agb_file = '%sAvitabile_AGB_%s_1km.tif' % (path2agb,country_code)
    agb = xr.open_rasterio(agb_file)
    agb_mask = agb.values[0]!=np.float32(agb.nodatavals[0])

    #create the land mask by combining the nodata masks for all data sources
    landmask = (wc2_mask.values & soil_mask.values & agb_mask)

    #create the empty array to store the predictors
    predictors = np.zeros([landmask.sum(),soil.shape[0]+wc2.shape[0]])

    # check the mask dimensions
    if len(landmask.shape)>2:
        print('\t\t caution shape of landmask is: ', landmask.shape)
        landmask = landmask[0]

    #iterate over variables to create the large array with data
    counter = 0
    #first wc2
    for bi in wc2:
        predictors[:,counter] = bi.values[landmask]
        counter += 1

    #then soil properties
    for sp in soil:
        predictors[:,counter] = sp.values[landmask]
        counter += 1
    print('Extracted WorldClim2 and SOILGRIDS data')

    agb_out = agb.values[landmask]

    return(predictors,agb_out,landmask)
