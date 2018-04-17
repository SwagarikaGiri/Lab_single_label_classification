import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import log_loss
import time
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
def readcsv_convert_list(input_file):
	dataset = pd.read_csv(input_file)
	csv_list = dataset.iloc[0:,0:14].values
	label_list=dataset.iloc[0:,14].values
	return csv_list,label_list
def get_kth_fold(kf,dataset,kth_fold,label_list):
	val=0
	train_=[]
	label_train=[]
	test_=[]
	label_test=[]
	for train_index,test_index in kf.split(dataset):
		if val==kth_fold:
			print train_index
			print test_index
			for i in range(0,len(train_index)):
				train_.append(dataset[train_index[i]])
				label_train.append(label_list[train_index[i]])
			for i in range(0,len(test_index)):
				test_.append(dataset[test_index[i]])
				label_test.append(label_list[test_index[i]])
			break
		val=val+1

	return train_,test_,label_train,label_test
def apply_svm(train_,label_train,test_):
	clf = svm.SVC()
	clf.fit(train_, label_train) 
	test_predict = clf.predict(test_)
	return test_predict

def write_file(label_file,predict_file,output_csv,predict_dec_tree,pred_naive_bayes,pred_random_forest):
	with open(output_csv,'wb') as output_csvfile:
		spamwriter = csv.writer(output_csvfile, delimiter=",",quotechar='|')
		spamwriter.writerow(["svm","decision tree","naive_bayes","random_forest"])
		for i in range(0,len(label_file)):
			string=str(label_file[i])+"_"+str(predict_file[i])+","+str(label_file[i])+"_"+str(predict_dec_tree[i])
			svm=str(label_file[i])+"_"+str(predict_file[i])
			dec=str(label_file[i])+"_"+str(predict_dec_tree[i])
			naive_bayes=str(label_file[i])+"_"+str(pred_naive_bayes[i])
			random_forest=str(label_file[i])+"_"+str(pred_naive_bayes[i])
			spamwriter.writerow([svm,dec,naive_bayes,random_forest])
def apply_decision_tree(train_,label_train,test_):
	clf = tree.DecisionTreeClassifier()
	clf.fit(train_, label_train) 
	test_predict = clf.predict(test_)
	return test_predict

def apply_naive_bayes(train_,label_train,test_):
	NB_model = GaussianNB()
	NB_model.fit(train_,label_train)
	NB_predict = NB_model.predict(test_)
	return NB_predict

def apply_random_forest(train_,label_train,test_):
	RF_model = RandomForestClassifier()
	RF_model.fit(train_,label_train)
	RF_predict = RF_model.predict(test_)
	return RF_predict

def create_result_for_k_fold():
	input_file="eeg_for.csv"
	csv_list,label_list=readcsv_convert_list(input_file)
	k_fold_var=int(raw_input("enter the k fold you need \t"))
	# k_fold_var=4
	kf = KFold(n_splits=k_fold_var)
	count=1
	for val in range(0,k_fold_var):
		train_,test_,label_train,label_test=get_kth_fold(kf,csv_list,val,label_list)
		print "applying svm \n"
		start_time = time.time()
		predict_test=apply_svm(train_,label_train,test_)
		print "svm complete \n"
		print("--- %s seconds --- for svm\n" % (time.time() - start_time))
		svm_time=time.time()-start_time
		print "applying decision tree\n"
		start_time = time.time()
		predict_dec_tree=apply_decision_tree(train_,label_train,test_)
		print("--- %s seconds --- for decision tree\n" % (time.time() - start_time))
		decision_tree_time=time.time()-start_time
		print "applying naive bayes \n"
		start_time = time.time()
		pred_naive_bayes=apply_naive_bayes(train_,label_train,test_)
		print("--- %s seconds --- for decision tree\n" % (time.time() - start_time))
		naive_bayes_time = time.time() - start_time
		print "applying random forest \n"
		start_time = time.time()
		pred_random_forest=apply_random_forest(train_,label_train,test_)
		print("--- %s seconds --- for decision tree\n" % (time.time() - start_time))
		random_forest_time = time.time() - start_time
		print " We have \t "+str(k_fold_var)+"\t folds to be calculated and this is\t"+str(val+1)+"\tfold"
		# result_file=raw_input('enter the file name for this fold in form of 1_3fold.csv form\t')
		result_file=str(count)+"_"+str(k_fold_var)+"fold.csv"
		count=count+1
		write_file(label_test,predict_test,result_file,predict_dec_tree,pred_naive_bayes,pred_random_forest)
	file="timing_"+str(k_fold_var)+"_fold.csv"
	with open(file,'wb') as output_csvfile:
		spamwriter = csv.writer(output_csvfile, delimiter=",",quotechar='|')
		spamwriter.writerow(["svm",str(svm_time)])
		spamwriter.writerow(["decision_tree",str(decision_tree_time)])
		spamwriter.writerow(["naive bayes",str(naive_bayes_time)])
		spamwriter.writerow(["random forest",str(random_forest_time)])

create_result_for_k_fold()


	


