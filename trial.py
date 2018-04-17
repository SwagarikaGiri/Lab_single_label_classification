import numpy as np
def rmse(y_predicted, y_actual):
	return ((y_predicted-y_actual)**2)
print rmse(1,0)
   
import time
start_time = time.time()
print start_time
print("--- %s seconds ---" % (time.time() - start_time))