import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4],[5,6],[9,0],[0,8],[7,8],[8,9]])
kf = KFold(n_splits=5)
# print kf.split(X)

for train_index,test_index in kf.split(X):
	print train_index,test_index


	
# print kf.get_n_splits(X

