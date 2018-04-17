import csv
def create_training_testing(input_csv,delimit):
	part1=[]
	part2=[]
	part3=[]
	part4=[]
	count=0
	with open(input_csv, 'rb') as input_csvfile:
			spamreader = csv.reader(input_csvfile, delimiter=delimit, quotechar='|')
			for line in spamreader:
				count=count+1
				if count%4==0:
					part1.append(line)
				if count%4==1:
					part2.append(line)
				if count%4==2:
					part3.append(line)
				if count%4==3:
					part4.append(line)
	return part1,part2,part3,part4
part1,part2,part3,part4=create_training_testing('eeg_for.csv',',')
print part1
print len(part1)