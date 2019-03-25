"""
general_plots
--------------------------------------------------------------------------------
GENERATE PLOTS FOR WORKSHOP
David T. Milodowski, 25/03/2019
"""

"""
import libraries needed
"""
import numpy as np
import matplotlib.pyplot as plt     # plotting package
import seaborn as sns               # another useful plotting package

"""
Part 2: Random forests
"""
# Figure 1 simple cross plot of three test datasets
def plot_test_data(X1,y1,X2,y2,X3,y3,show=True):
    fig,axes = plt.subplots(nrows=1,ncols=3,figsize = (8,3))
    axes[0].plot(X1,y1,'.')
    axes[1].plot(X2,y2,'.')
    axes[2].plot(X3,y3,'.')
    axes[0].set_xlim((0,10));axes[0].set_ylim((-5,28))
    axes[2].set_xlim((0,10));axes[2].set_ylim((-5,28))
    fig1.tight_layout()
    if show:
        fig1.show()
    return fig1,axes

# Figure 2 adding regression results
def plot_test_data_with_regression_results(X1,y1,X2,y2,X3,y3,X_test,y1_test,y1_test_lm,y2_test,y3_test,y3_test_lm,show=True):
    fig2,axes = plt.subplots(nrows=1,ncols=3,figsize = (8,3))
    axes[0].plot(X,y1,'.',label='data',color='0.5')
    axes[0].plot(X_test,y1_test,'-',color='red',label='naive rf model')
    axes[0].plot(X_test,y1_test_lm,'-',color='blue',label='linear regression')
    axes[1].plot(X,y2,'.',color='0.5')
    axes[1].plot(X_test,y2_test,'-',color='red')
    axes[2].plot(X3,y3,'.',color='0.5')
    axes[2].plot(X_test,y3_test,'-',color='red')
    axes[2].plot(X_test,y3_test_lm,'-',color='blue')
    axes[0].set_xlim((0,10));axes[0].set_ylim((-5,28))
    axes[2].set_xlim((0,10));axes[2].set_ylim((-5,28))
    axes[0].legend(loc='lower right',fontsize = 8)
    fig2.tight_layout()
    if show:
        fig2.show()
    return fig2,axes
