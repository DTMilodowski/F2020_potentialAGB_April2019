"""
30/11/2018 - DTM
Create training mask for RFR routine

"""
import numpy as np
import xarray as xr
import glob
def set(path2root='../',subset=1):
    # subset 1: HFL and ESACCI)
    if subset == 1:
        lcfiles = sorted(glob.glob('%s/data/landcover/*lccs-class*tif' % path2root))
        lc = xr.open_rasterio(lcfiles[0]).values[0]
        forestmask=np.ones(lc.shape)
        sparsemask=np.ones(lc.shape)

        sparsemask = (lc>=120)*(lc<130) + (lc>=140)*(lc<160) + (lc>=200)*(lc<210) # shrub, lichen, sparse, bare
        #sparsemask = (lc>=140)*(lc<160) + (lc>=200)*(lc<210)# lichen, sparse, bare
        forestmask =(lc>=50)*(lc<=90) + (lc==160)  + (lc==170)

        # loop through years and only keep pixels tha do not change from year to year
        lc_start = 1993
        lc_end = 2015
        for ff in range(len(lcfiles)):
            lc_p = lc.copy()
            lc = xr.open_rasterio(lcfiles[ff]).values[0]
            forestmask *= (lc==lc_p)
            sparsemask *= (lc==lc_p)
        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/data/landcover/*HFL*tif' % path2root)[0]).values[0]
        forestmask=forestmask*(hfl==1)
        # merge masks
        training_mask=sparsemask+forestmask

    # subset 2: HFL only
    elif subset == 2:
        # load hinterland forests
        hfl = xr.open_rasterio(glob.glob('%s/data/landcover/*HFL*tif' % path2root)[0]).values[0]
        training_mask=(hfl==1)

    # subset 3: a bespoke dataset for the workshop
    elif subset == 3:
        training = xr.open_rasterio(glob.glob('%s/data/training/*tif' % path2root)[0]).values[0]
        training_mask=(training>0)
    return training_mask
