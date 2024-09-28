#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""avg_metric: avergae scores for all relations"""

__author__ = "Ashutosh Bajpai"
__copyright__ = ""
__license__ = ""
__project__= "TeCFaP"
__version__ = "1.0.0"
__maintainer__ = "Ashutosh Bajpai"

#_______________________________________________________________________________________________________________


import sys,os
import pandas as pd
import json
import csv
import numpy as np
import itertools 
import csv

nature = "strict"

# metric scores input file
input_file = sys.argv[1]


relation_id =[]
avg_accuracy =[]
forward_accuracy =[]
backward_accuracy=[]
avg_cons=[]
forward_cons=[]
backward_cons=[]
avg_cons_acc=[]
forward_cons_acc=[]
backward_cons_acc=[]
avg_succ_pats=[]
forward_succ_pats=[]
backward_succ_pats=[]
avg_succ_vsub=[]
forward_succ_vsub=[]
backward_succ_vsub=[]


avg_unk_cons=[]
forward_unk_cons=[]
backward_unk_cons=[]
avg_know_cons=[]
forward_know_cons=[]
backward_know_cons=[]

with open(input_file,"r") as f:
		csvreader = csv.reader(f,delimiter=',')
		next(csvreader)
		check =0
		for row in csvreader:
			if len(row)>9:
				relation_id.append(float(row[0]))
				avg_accuracy.append(float(row[1]))
				forward_accuracy.append(float(row[2]))
				backward_accuracy.append(float(row[3]))
				avg_cons.append(float(row[4]))
				forward_cons.append(float(row[5]))
				backward_cons.append(float(row[6]))
				avg_cons_acc.append(float(row[7]))
				forward_cons_acc.append(float(row[8]))
				backward_cons_acc.append(float(row[9]))
				avg_succ_pats.append(float(row[10]))
				forward_succ_pats.append(float(row[11]))
				backward_succ_pats.append(float(row[12]))
				avg_succ_vsub.append(float(row[13]))
				forward_succ_vsub.append(float(row[14]))
				backward_succ_vsub.append(float(row[15]))
			elif len(row)>2:
				if check==0:
					check =1
					continue
				print('avg_unk_cons',round(float(row[0]),4))
				print('forward_unk_cons',round(float(row[1]),4))
				print('backward_unk_cons',round(float(row[2]),4))
				print()
				print('avg_know_cons',round(float(row[3]),4))
				print('forward_know_cons',round(float(row[4]),4))
				print('backward_know_cons',round(float(row[5]),4))

print(sum(avg_accuracy))
print()
print('avg_accuracy',round(sum(avg_accuracy)/len(avg_accuracy),4))
print('forward_accuracy',round(sum(forward_accuracy)/len(forward_accuracy),4))
print('backward_accuracy',round(sum(backward_accuracy)/len(backward_accuracy),4))
print()
print('avg_cons',round(sum(avg_cons)/len(avg_cons),4))
print('forward_cons',round(sum(forward_cons)/len(forward_cons),4))
print('backward_cons',round(sum(backward_cons)/len(backward_cons),4))
print()
print('avg_cons_acc',round(sum(avg_cons_acc)/len(avg_cons_acc),4))
print('forward_cons_acc',round(sum(forward_cons_acc)/len(forward_cons_acc),4))
print('backward_cons_acc',round(sum(backward_cons_acc)/len(backward_cons_acc),4))
print()
print('avg_succ_pats',round(sum(avg_succ_pats)/len(avg_succ_pats),4))
print('forward_succ_pats',round(sum(forward_succ_pats)/len(forward_succ_pats),4))
print('backward_succ_pats',round(sum(backward_succ_pats)/len(backward_succ_pats),4))
print()
print('avg_succ_vsub',round(sum(avg_succ_vsub)/len(avg_succ_vsub),4))
print('forward_succ_vsub',round(sum(forward_succ_vsub)/len(forward_succ_vsub),4))
print('backward_succ_vsub',round(sum(backward_succ_vsub)/len(backward_succ_vsub),4))


