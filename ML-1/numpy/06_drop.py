import numpy as np

a = np.array([1,2,3,np.nan,5,6,7,np.nan])


print(a[~np.isnan(a)])