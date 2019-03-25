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
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

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

# Figure 3, basic cal-val example
def plot_cal_val(y_train,y_train_rf,y_test,y_test_rf,show=True):
    calval_df = pd.DataFrame(data = {'val_obs': y_test,
                                     'val_model': y_test_rf,
                                     'cal_obs': y_train,
                                     'cal_model': y_train_rf})

    fig3,axes= plt.subplots(nrows=1,ncols=2,figsize=[8,4])
    sns.regplot(x='cal_obs',y='cal_model',data=calval_df,marker='+',
                truncate=True,ci=None,ax=axes[0])
    axes[0].annotate('calibration R$^2$ = %.02f\nRMSE = %.02f' %
                (r2_score(y_train,y_train_rf),np.sqrt(mean_squared_error(y_train,y_train_rf))),
                xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='left', verticalalignment='top')
    sns.regplot(x='val_obs',y='val_model',data=calval_df,marker='+',
                truncate=True,ci=None,ax=axes[1])
    axes[1].annotate('validation R$^2$ = %.02f\nRMSE = %.02f'
                % (r2_score(y_test,y_test_rf),np.sqrt(mean_squared_error(y_test,y_test_rf))),
                xy=(0.05,0.95), xycoords='axes fraction',backgroundcolor='none',
                horizontalalignment='left', verticalalignment='top')
    axes[0].axis('equal')
    axes[1].axis('equal')
    fig3.tight_layout()
    if show:
        fig3.show()
    return fig3,axes
