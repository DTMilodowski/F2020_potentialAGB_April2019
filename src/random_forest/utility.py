"""
utility
--------------------------------------------------------------------------------
Miscellaneous useful functions
David T. Milodowski, 25/03/2019
"""
import numpy as np

"""
generate_simple_test_data
-----------------------------------------
Genereate three contrasting datasets on which to conduct simple tests
"""
def generate_simple_test_data():
    X1 = np.random.random(500)*10. # generate a set of random numbers between 0 and 10
    y1= 2.3*X1 + np.random.randn(500)*1. # generate a linear relationship with some noise - here noise comes from a random distribution with mean 0 and standard deviation of 1.
    X2=X1.copy()
    y2= np.cos(X2**0.5*5)+0.2*X2 + np.random.randn(500)*0.3 # a more complex nonlinear function
    # Finally, lets make a version of y1 where we have gaps at the start, end and
    # middle of the dataset. Don't worry about the details for now
    temp = X1.copy()
    temp[X1<1.7] = np.nan; temp[X1>9.5]=np.nan; temp[np.all((X1>3.4,X1<6.5),axis=0)]=np.nan
    X3 = X1[np.isfinite(temp)]
    y3=y1[np.isfinite(temp)]
    return X1,y1,X2,y2,X3,y3
