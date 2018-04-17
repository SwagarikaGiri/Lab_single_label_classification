"""here  we will calculate the metrix"""
""" this is for svm"""
import csv
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from math import log
from math import sqrt
from numpy import array
print "THIS IS FOR SVM \n"
confusion_matrix=dict()
def logloss(true_label, predicted ):
	eps=1e-15
	p = np.clip(predicted, eps, 1 - eps)
	if true_label == 1:
	  return -log(p)
	else:
	  return -log(1 - p)
def mean_squared_error(y_actual,y_predicted):
	return ((y_predicted-y_actual)**2)
def seperate_label_predict(input_csv):
	log_loss_=0
	mean_square_error_=0
	count=0

	with open(input_csv, 'rb') as input_csvfile:
		spamreader = csv.reader(input_csvfile, delimiter=",", quotechar='|')
		for line in spamreader:
			count=count+1
			if count !=1:
				val=line[0]
				list_=val.split("_")
				true_label=int(list_[0])
				predicted=int(list_[1])
				log_loss_=log_loss_+logloss(true_label,predicted)
				mean_square_error_=mean_square_error_+mean_squared_error(true_label, predicted)
		mean_square_error_=float(mean_square_error_)/float(count)
		log_loss_=float(log_loss_)/float(count)
	return log_loss_,mean_square_error_

def create_confusionmatrix(input_csv):
	confusion_matrix.clear()
	confusion=dict()
	log_loss=0
	mean_sqr_error=0
	count=0
	with open(input_csv, 'rb') as input_csvfile:
		spamreader = csv.reader(input_csvfile)
		for line in spamreader:
			count=count+1
			if count !=1:
				val=line[0]
				list_=val.split("_")
				if (list_[0],list_[1]) not in confusion:
					confusion[list_[0],list_[1]]=1
				else:
					confusion[list_[0],list_[1]]=confusion[list_[0],list_[1]]+1
		print confusion
		states=['0','1']
		for i in states:
			for j in states:
				if (i,j) in confusion:
					confusion_matrix[i,j]=confusion[i,j]
				else:
					confusion_matrix[i,j]=0
	return confusion_matrix
def metrix(lexicon):
	TP=0
	TN=0
	FP=0
	FN=0
	TP=lexicon['1','1']
	TN=lexicon['0','0']
	FP=lexicon['0','1']
	FN=lexicon['1','0']
	total=TP+TN+FP+FN
	prec_denom=TP+FP
	recall_denom=TP+FN
	if prec_denom==0:
		precision=0
	else:
		precision=float(TP)/float(prec_denom)
	if recall_denom==0:
		recall=0
	else:
		recall=float(TP)/float(recall_denom)
	f1_denom=precision+recall
	if f1_denom==0:
		f1_score=0
	else:
		f1_score=float(2*precision*recall)/float(precision+recall)
	accuracy=float(TP)/total
	error_rate = 1-accuracy
	return TP,TN,FP,FN,precision,recall,f1_score,accuracy,error_rate



K=int(raw_input("enter the K fold value\t"))

avg_precision=0
avg_recall=0
avg_f1_score=0
avg_accuracy=0
avg_log_loss=0
avg_error_rate=0
avg_mean_square_error=0
lexicon=dict()
for i in range(1,K+1):
	lexicon.clear()
	str_=""
	str_=str(i)+"_"+str(K)+"fold.csv"
	lexicon=create_confusionmatrix(str_)
	log_loss,mean_sqr_error=seperate_label_predict(str_)
	TP,TN,FP,FN,precision,recall,f1_score,accuracy,error_rate=metrix(lexicon)
	avg_precision=avg_precision+precision
	avg_recall=avg_recall+recall
	avg_f1_score=avg_f1_score+f1_score
	avg_accuracy=avg_accuracy+accuracy
	avg_log_loss=avg_log_loss+log_loss
	avg_mean_square_error=avg_mean_square_error+mean_sqr_error
	avg_error_rate=avg_error_rate+error_rate
avg_precision=float(avg_precision)/K
avg_recall=float(avg_recall)/K
avg_f1_score=float(avg_f1_score)/K
avg_accuracy=float(avg_accuracy)/K
avg_log_loss=float(avg_log_loss)/K
avg_mean_square_error=float(avg_mean_square_error)/K
avg_error_rate=float(avg_error_rate)/K
print "average precison: "+str(avg_precision)+"\taverage recall: "+str(avg_recall)+"\taverage f1 score:"+str(avg_f1_score)+"\t average accuracy:"+str(avg_accuracy)
print "avg log loss:"+str(avg_log_loss)+"\t avg mean square error: "+str(avg_mean_square_error)+"\t avg error rate: "+str(avg_error_rate)
file="svm_"+str(K)+"_fold.csv"
with open(file,'wb') as output_csvfile:
			spamwriter = csv.writer(output_csvfile, delimiter=",",quotechar='|')
			spamwriter.writerow(["metric","score"])
			spamwriter.writerow(["average precison:",str(round(avg_precision,2))])
			spamwriter.writerow(["average recall:",str(round(avg_recall,2))])
			spamwriter.writerow(["average f1 score:",str(round(avg_f1_score,2))])
			spamwriter.writerow(["average accuracy:",str(round(avg_accuracy,2))])
			spamwriter.writerow(["avg log loss:",str(round(avg_log_loss,2))])
			spamwriter.writerow(["avg mean square error: ",str(round(avg_mean_square_error,2))])
			spamwriter.writerow(["avg error rate"+str(round(avg_error_rate,2))])
			

	