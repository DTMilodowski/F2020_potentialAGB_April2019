"""

regrid data sets
- this routine clips and regrids the data layers to be fed into the RFR analysis to estimate potential biomass
- it predominately makes use of gdal
- the 30 arcsec worldclim2 bio data are used as the template onto which other data are regridded
- regridding modes are as follows:
  - AGB (Avitabile) - nearest neighbour
  - ESA CCI landcover - mode
  - Primary forest maps (Morgano et al.) - optional
  - Intact forest landscapes (Popatov et al.) - optional
  - SOILGRIDS - nearest neighbour
- the extent is specified based on the bounding box (lat long limits)
- if required a mask is generated to clip the output to the specified national boundaries

- the regridding routines utilise the following packages:
  - GDAL
  This is run from the command line, rather than with the python bindings

14/09/2018
David T. Milodowski

"""

# This script just calls GDAL and NCO to clip existing datasets to a specified bounding box
import os
import glob

# Bounding Box
W = -79.
E = -51.
N = 13.
S = -5.

res = 0.008333333333333*2 # approx 2km resolution

# Start with the worldclim2 data. This should be straightforward using gdal as are just clipping
# to the target extent
wc2dir = '/disk/scratch/local.2/worldclim2/'
wc2files = glob.glob('%swc2*tif' % wc2dir);wc2files.sort()
wc2subset = []
wc2vars = []
for ff,fname in enumerate(wc2files):
    variable = fname.split('_')[-1].split('.')[0]
    outfname = fname.split('/')[-1][:-4]
    #if int(variable) in ['01','04','05','06','12','13','14','15']:
    if int(variable) in range(1,20):
        wc2vars.append(variable)
        wc2subset.append(fname)
        os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r average  %s\
                    ../../data/climatology/worldclim2/colombia_%s_2km.tif" % (W,S,E,N,res,res,wc2files[ff],outfname))

# Soilgrids next. Soilgrids are downloaded as geotiffs. Resolution is identical to worldclim, so regridding process is the same
sgdir = '/disk/scratch/local.2/soilgrids/1km/'
sgfiles = glob.glob('%s*_1km_ll.tif' % sgdir);sgfiles.sort()
sgsubset = []
sgvars = []
for ff,fname in enumerate(sgfiles):
    variable = fname.split('/')[-1][:-11]
    outfname = fname.split('/')[-1][:-11]
    sgvars.append(variable)
    sgsubset.append(fname)
    os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r average %s ../../data/soils/colombia_%s_2km.tif" % (W,S,E,N,res,res,sgfiles[ff],outfname))

# Aboveground Biomass - Avitabile map - 1 km resolution so same gdal example should be sufficient
agbfiles= ['/home/dmilodow/DataStore_GCEL/AGB/avitabile/Avitabile_AGB_Map/Avitabile_AGB_Map.tif','/home/dmilodow/DataStore_GCEL/AGB/avitabile/Avitabile_AGB_Uncertainty/Avitabile_AGB_Uncertainty.tif']
agbvars = ['Avitabile_AGB','Avitabile_AGB_Uncertainty']
os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r average  %s\
            ../../data/agb/colombia_%s_2km.tif" % (W,S,E,N,res,res,agbfiles[0],agbvars[0]))
os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r average  %s\
            ../../data/agb/colombia_%s_2km.tif" % (W,S,E,N,res,res,agbfiles[1],agbvars[1]))

# ESA CCI landcover - the original resolution of these rasters is higher than the reference data, so this needs to be regridded. The mode landcover class will be used.
# Note that the ESA CCI landcover data are originally in netcdf format.
lcdir = '/home/dmilodow/DataStore_GCEL/ESA_CCI_landcover/'
lcfiles = glob.glob('%s*.nc' % lcdir);lcfiles.sort()
for ff,fname in enumerate(lcfiles):
    outfname = fname.split('/')[-1][:22]+fname.split('/')[-1][27:42]+"-2km-mode-lccs-class"
    os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r mode -of GTIFF\
                NETCDF:%s:lccs_class ../../data/landcover/colombia-%s_2km.tif" % (W,S,E,N,res,res,lcfiles[ff],outfname))

# Hinterland forest landscapes (Potapov et al.)
hfl_file = '/disk/scratch/local.2/global_hinterland_forests/source_files/SAM_2013hinterland_25vcf.tif'
os.system("gdalwarp -overwrite -te %f %f %f %f -dstnodata -9999  -tr %f -%f\
            -r mode -of GTIFF %s ../../data/landcover/colombia_HFL_2013_2km.tif" % (W,S,E,N,res,res,hfl_file))


# WORLDCLIM scenarios
rcps = ['26','45','60','85']
wcdir = '/home/dmilodow/DataStore_GCEL/WorldClim1_4/'
for rcp in rcps:
    wcfiles = glob.glob('%srcp%s_30s/he*tif' % (wcdir,rcp));wcfiles.sort()
    wcsubset = []
    for ff,fname in enumerate(wcfiles):
        variable = fname.split('/')[-1].split('.')[0]
        temp = fname.split('/')[-1][:-4]
        outfname = temp[:6]+'_'+temp[-2:]
        wcsubset.append(fname)
        os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r average  %s\
                    ../../data/scenarios/rcp%s/colombia_%s_2km.tif" % (W,S,E,N,res,res,wcfiles[ff],rcp,outfname))

# WORLDCLIM 1.4
wcfiles = glob.glob('%s/current/bio*.bil' % wcdir);wcfiles.sort()
wcsubset = []
for ff,fname in enumerate(wcfiles):
    variable = fname.split('/')[-1].split('.')[0]
    wcsubset.append(fname)
    os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r average  %s\
                    ../../data/climatology/worldclim1_4/colombia_%s_2km.tif" % (W,S,E,N,res,res,wcfiles[ff],variable))

# Bespoke training set
file = '/home/dmilodow/DataStore_DTM/FOREST2020/PotentialBiomassRFR/output/COL_003_AGB_potential_RFR_worldclim_soilgrids.nc'
os.system("gdalwarp -overwrite -te %f %f %f %f -tr %f -%f -r mode -of GTIFF\
                NETCDF:%s:trainset3 ../../data/training/colombia_trainset_2km.tif" % (W,S,E,N,res,res,file))
