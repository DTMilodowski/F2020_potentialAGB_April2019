import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal
import os
import osr
import sys
import xarray as xr

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib import rcParams
# Set up some basiic parameters for the plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['arial']
rcParams['font.size'] = 9
rcParams['legend.numpoints'] = 1
axis_size = rcParams['font.size']

# Convert a python array with float variables into a three-band RGB array with the colours specified
# according to a given colormap and upper and lower limits to the colormap
def convert_array_to_rgb(array, cmap, ulim, llim, nodatavalue=-9999):
    norm = mpl.colors.Normalize(vmin=llim, vmax=ulim)
    rgb_array= cm.ScalarMappable(norm=norm,cmap=cmap).to_rgba(array)[:,:,:-1]*255
    mask = np.any((~np.isfinite(array),array==nodatavalue),axis=0)
    rgb_array[mask,:]=np.array([255.,0.,255.])
    return rgb_array

# create geotransformation object from xarray- needed to georeference geotiff
def create_geoTrans(array,x_name='longitude',y_name='latitude'):
    lat = array.coords[y_name].values
    lon = array.coords[x_name].values
    dlat = lat[1]-lat[0]
    dlon = lon[1]-lon[0]
    geoTrans = [0,dlon,0,0,0,dlat]
    geoTrans[0] = np.min(lon)
    if geoTrans[5]>0:
        geoTrans[3]=np.min(lat)
    else:
        geoTrans[3]=np.max(lat)
    return geoTrans

# check orientations are correct for displaying in GIS programs
def check_array_orientation(array,geoTrans,north_up=True):
    if north_up:
        # for north_up array, need the n-s resolution (element 5) to be negative
        if geoTrans[5]>0:
            geoTrans[5]*=-1
            geoTrans[3] = geoTrans[3]-(array.shape[0]+1.)*geoTrans[5]
        # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
        if len(array.shape) < 2:
            print('array has less than two dimensions! Unable to write to raster')
            sys.exit(1)
        elif len(array.shape) == 2:
            array = np.flipud(array)
        elif len(array.shape) == 3:
            (NRows,NCols,NBands) = array.shape
            for i in range(0,NBands):
                array[:,:,i] = np.flipud(array[:,:,i])
        else:
            print('array has too many dimensions! Unable to write to raster')
            sys.exit(1)

    else:
        # for north_up array, need the n-s resolution (element 5) to be positive
        if geoTrans[5]<0:
            geoTrans[5]*=-1
            geoTrans[3] = geoTrans[3]-(array.shape[0]+1.)*geoTrans[5]
        # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
        if len(array.shape) < 2:
            print('array has less than two dimensions! Unable to write to raster')
            sys.exit(1)
        elif len(array.shape) == 2:
            array = np.flipud(array)
        elif len(array.shape) == 3:
            (NRows,NCols,NBands) = array.shape
            for i in range(0,NBands):
                array[:,:,i] = np.flipud(array[:,:,i])
        else:
            print ('array has too many dimensions! Unable to write to raster')
            sys.exit(1)

    # Get array dimensions and flip so that it plots in the correct orientation on GIS platforms
    if len(array.shape) < 2:
        print ('array has less than two dimensions! Unable to write to raster')
        sys.exit(1)
    elif len(array.shape) == 2:
        array = np.flipud(array)
    elif len(array.shape) == 3:
        (NRows,NCols,NBands) = array.shape
        for i in range(0,NBands):
            array[:,:,i] = np.flipud(array[:,:,i])
    else:
        print ('array has too many dimensions! Unable to write to raster')
        sys.exit(1)

    return array,geoTrans


# Function to write an EO lab data layer from an array
# For the moment, this can only accept an input 2D xarray with dimensions
# latitude and longitude
def write_array_to_data_layer_GeoTiff(array,geoTrans, OUTFILE_prefix, EPSG_CODE='4326',
                                        north_up=True):
    NBands = 1
    NRows,NCols = array.values.shape

    # create geotrans object
    geoTrans = create_geoTrans(array)

    # check orientation
    array.values,geoTrans = check_array_orientation(array.values,geoTrans,north_up=north_up)

    # set nodatavalue
    array.values[np.isnan(array.values)] = -9999

    # Write GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    # set all the relevant geospatial information
    dataset = driver.Create( OUTFILE_prefix+'_data.tif', NCols, NRows, NBands, gdal.GDT_Float32 )
    dataset.SetGeoTransform( geoTrans )
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS( 'EPSG:'+EPSG_CODE )
    dataset.SetProjection( srs.ExportToWkt() )
    # write array
    dataset.GetRasterBand(1).SetNoDataValue( -9999 )
    dataset.GetRasterBand(1).WriteArray( array.values )
    dataset = None
    return 0


# This function is similar to before, except that now it writes two GeoTIFFs - a data layer and a
# display layer.  For the moment, this can only accept an input 2D xarray with dimensions
# latitude and longitude
def write_array_to_display_layer_GeoTiff(array, OUTFILE_prefix, cmap, ulim, llim,
                                        EPSG_CODE_DATA='4326', EPSG_CODE_DISPLAY='3857',
                                        north_up=True):
    NBands = 1
    NBands_RGB = 3
    NRows,NCols = array.values.shape

    # get geotransform
    geoTrans = create_geoTrans(array)
    # check orientation of map
    array.values,geoTrans = check_array_orientation(array.values,geoTrans,north_up=north_up)

    # set nodata as -9999
    array.values[np.isnan(array.values)] = -9999

    # Convert RGB array
    rgb_array = convert_array_to_rgb(array.values,cmap,ulim,llim)

    # Write Data Layer GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    # set all the relevant geospatial information
    dataset = driver.Create( OUTFILE_prefix+'_data.tif', NCols, NRows, NBands, gdal.GDT_Float32 )
    dataset.SetGeoTransform( geoTrans )
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS( 'EPSG:'+EPSG_CODE_DATA )
    dataset.SetProjection( srs.ExportToWkt() )
    # write array
    dataset.GetRasterBand(1).SetNoDataValue( -9999 )
    dataset.GetRasterBand(1).WriteArray( array.values )
    dataset = None

    # Write Display Layer GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    # set all the relevant geospatial information
    dataset = driver.Create( 'temp.tif', NCols, NRows, NBands_RGB, gdal.GDT_Byte )
    dataset.SetGeoTransform( geoTrans )
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS( 'EPSG:'+EPSG_CODE_DATA )
    dataset.SetProjection( srs.ExportToWkt() )
    # write array
    for bb in range(0,3):
        dataset.GetRasterBand(bb+1).WriteArray( rgb_array[:,:,bb] )
    dataset = None

    # now use gdalwarp to reproject
    os.system("gdalwarp -t_srs EPSG:" + EPSG_CODE_DISPLAY + " temp.tif " + OUTFILE_prefix+'_display.tif')
    os.system("rm temp.tif")
    return 0


# A function to produce a simple map legend for quantitative data layers
def plot_legend(cmap,ulim,llim,axis_label, OUTFILE_prefix,extend='neither'):
    norm = mpl.colors.Normalize(vmin=llim, vmax=ulim)
    #plt.figure(1, facecolor='White',figsize=[2, 1])
    fig,ax = plt.subplots(facecolor='White',figsize=[2, 1])
    ax = plt.subplot2grid((1,1),(0,0))
    cb = mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,orientation='horizontal',extend=extend)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.set_label(axis_label,fontsize = axis_size)
    fig.tight_layout()
    fig.savefig(OUTFILE_prefix+'_legend.png')
    return 0


def plot_legend_listed(cmap,labels,axis_label, OUTFILE_prefix):
    bounds = np.arange(len(labels)+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    #plt.figure(1, facecolor='White',figsize=[1.5, 1])
    fig,ax = plt.subplots(facecolor='White',figsize=[1.5, 1])
    ax = plt.subplot2grid((1,1),(0,0))
    cb = mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm,
                                orientation='vertical')
    n_class = labels.size
    loc = np.arange(0,n_class)+0.5
    cb.set_ticks(loc)
    cb.set_ticklabels(labels)
    cb.update_ticks()

    ax.set_title(axis_label,fontsize = axis_size)
    fig.tight_layout()
    fig.savefig(OUTFILE_prefix+'_legend.png')
    return 0
