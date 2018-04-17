from sklearn.metrics import log_loss, mean_squared_error
from math import sqrt

import pandas
import numpy
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
dataframe = pandas.read_csv('train.txt', delimiter = ",", names =["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4","eyeDetection"])
column_name = dataframe.columns.values
#print(column_name)
data_c = dataframe.loc[:,column_name[0:len(column_name)-1]]
data_d = dataframe.iloc[:,-1]
#print(tra_column)
#print(test_column)
(train_data_c, test_data_c, train_data_d, test_data_d) = train_test_split(data_c, data_d, test_size = 0.25) 
#print(test_data_c)
#print(type(test_data_d))
""" NAIVE BAYES CLASSIFICATION """
from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB()
NB_model.fit(train_data_c,train_data_d)
NB_predict = NB_model.predict(test_data_c)
NB_conf_matrix = confusion_matrix(test_data_d,NB_predict)

NB_accuracy = (NB_conf_matrix[0][0]+NB_conf_matrix[1][1])/numpy.sum(NB_conf_matrix)
#print(type(NB_predict))
print("\nNaive Bayes \n\t\taccuracy is ",NB_accuracy)
#print("\t\tconfusion matrix ",NB_conf_matrix)
x1 = log_loss(test_data_d, NB_predict)
rms1 = sqrt(mean_squared_error(test_data_d, NB_predict))
print("\t\tmean square error ",rms1)
print("\t\tlog loss ",x1,"\n")
""" DESION TREE CLASSIFICATION """

DecT_model = DecisionTreeClassifier()
DecT_model.fit(train_data_c,train_data_d)
DecT_predict = DecT_model.predict(test_data_c)
DecT_conf_matrix = confusion_matrix(test_data_d,DecT_predict)
#print(NB_conf_matrix)
DecT_accuracy = (DecT_conf_matrix[0][0]+DecT_conf_matrix[1][1])/numpy.sum(DecT_conf_matrix)
print("Decision Tree \n\t\taccuracy is ",DecT_accuracy)
rms2 = sqrt(mean_squared_error(test_data_d, DecT_predict))
print("\t\tmean square error  ",rms2)
x2 = log_loss(test_data_d, DecT_predict)
print("\t\tlog loss ",x2,"\n")

""" RANDOM FOREST CLASSIFIER """
RF_model = RandomForestClassifier()
RF_model.fit(train_data_c,train_data_d)
RF_predict = RF_model.predict(test_data_c)
RF_conf_matrix = confusion_matrix(test_data_d,RF_predict)
#print(NB_conf_matrix)
RF_accuracy = (RF_conf_matrix[0][0]+RF_conf_matrix[1][1])/numpy.sum(RF_conf_matrix)
print("Random Forest \n\t\taccuracy is ",RF_accuracy)
rms3 = sqrt(mean_squared_error(test_data_d, RF_predict))
print("\t\tmean square error ",rms3)
x3 = log_loss(test_data_d, RF_predict)
print("\t\tlog loss ",x3,"\n")


""" SUPPORT VECTOR MACHINE """
SV_model = SVC()
SV_model.fit(train_data_c,train_data_d)
SV_predict = SV_model.predict(test_data_c)
SV_conf_matrix = confusion_matrix(test_data_d,SV_predict)
#print(NB_conf_matrix)
SV_accuracy = (SV_conf_matrix[0][0]+SV_conf_matrix[1][1])/numpy.sum(SV_conf_matrix)
print("Support Vector \n\t\taccuracy is ",SV_accuracy)
rms4 = sqrt(mean_squared_error(test_data_d, SV_predict))
print("\t\tmean square error ",rms4)
x4 = log_loss(test_data_d, SV_predict)
print("\t\tlog loss ",x4)

print("\nAverage log loss is  ",(x1+x2+x3+x4)/4,"\n")