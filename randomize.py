import csv
list_=[]
with open("egg_for.csv", 'rb') as input_csvfile:
	spamreader = csv.reader(input_csvfile, delimiter=",", quotechar='|')
