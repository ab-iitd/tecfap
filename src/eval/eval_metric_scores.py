
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""eval_metric_scores: calculate meteric scores for model responses vs ground truth"""

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

projpath = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute().parent.absolute()
nature = "strict"
# model response file
input_file = sys.argv[1]
output_fine_name = "test_metrics_" + input_file
isTestRun = True # true if metric to be calculated for the response file generated from the test run indexes

if isTestRun:
    # reading test indexes
    dftest = pd.read_csv(os.path.join(projpath,'dataset/temp-cofac/test_index.csv'))
    test_index = dftest['test_index'].tolist()

dt = pd.read_csv(input_file, dtype=str)

clean_head = ['a', 'an', 'the', 's', '"', "'"]
pred_v = dt['pred_v_sub'].to_list()
vls = dt['v_sub'].to_list()
pred_v_tokens= [str(pv).replace(".","").strip().replace(",","").replace("</s>","").replace('"','').split() for pv in pred_v]

new_pred_v=[]
for i in range(len(pred_v_tokens)):
	if pred_v_tokens[i] != []:
		if pred_v_tokens[i][0] in clean_head: #or any(map(str.isdigit, pred_v_tokens[i][0])):
			new_pred_v.append(" ".join(pred_v_tokens[i][1:len(str(vls[i]).split())+1]))
		else:
			new_pred_v.append(" ".join(pred_v_tokens[i][:len(str(vls[i]).split())]))
	else:
		new_pred_v.append("NO_OUTPUT")

#pred_v   =[" ".join(pred_v_tokens[i][1:len(str(vls).split())+1]) if pred_v_tokens[i][0] in clean_head else " ".join(pred_v_tokens[i][:len(str(vls).split())]) for i in range(len(pred_v_tokens))]

#pred_v = [" ".join(str(pv).split()[:5]) for pv in pred_v]
dt['pred_v_sub'] = new_pred_v

print(dt.columns)

#### "a b c" and "a c b" score = 1/3
def calculate_all_accuracy_scores(df):
	scores = []
	values = df['v_sub'].to_list()
	pred_values = df['pred_v_sub'].to_list()
	for i in range(len(values)):
		word = values[i].split()
		pred_word = str(pred_values[i]).split()
		cnt = 0
		idx = 0

		for j in range(len(word)):
			if j<len(pred_word):
				if word[j] == pred_word[j]:
					cnt = cnt+1
			else:
				break

		scores.append((cnt*1.0)/len(word))
	return scores


def get_rel_consistencies(dt,unique_rel_id,direction):
	rel_const ={}
	for rel_id in unique_rel_id:
		df_sub = dt[(dt['relID']==rel_id) & (dt['patDir']==direction)]
		#print(df_sub)
		unique_entities_pair_id = list(set(df_sub['ent_pair_id'].to_list()))
		entity_const = {}
		for ent_pair in unique_entities_pair_id:
			df_sub1 = df_sub[(df_sub['ent_pair_id']==ent_pair)]
	
			lab = df_sub1['v_sub'].to_list()
			pred = df_sub1['pred_v_sub'].to_list()
			cons =0
			for k in range(len(lab)):
				for j in range(k+1,len(pred)):
					if pred[k] == pred[j]:
						cons += 1

			n = len(lab)-1
			if n==0:
				average_const = 0
			else:
				average_const = cons/((n*(n+1))/2)
			entity_const[ent_pair] = average_const
		if len(entity_const) >=1:
			rel_const[rel_id] = entity_const
	return rel_const


def get_cons_acc(dt,unique_rel_id,direction):

	rel_const_acc ={}
	for rel_id in unique_rel_id:
		df_sub = dt[(dt['relID']==rel_id) & (dt['patDir']==direction)]
		#print(df_sub)
		unique_entities_pair_id = list(set(df_sub['ent_pair_id'].to_list()))
		entity_const_acc = {}
		for ent_pair in unique_entities_pair_id:
			df_sub1 = df_sub[(df_sub['ent_pair_id']==ent_pair)]
	
			lab = df_sub1['v_sub'].to_list()
			pred = df_sub1['pred_v_sub'].to_list()
			scores = df_sub1['scores'].to_list()
			corr =0

			if len(list(set(scores)))==1:
				corr = scores[0]

			entity_const_acc[ent_pair] = corr
		rel_const_acc[rel_id] = entity_const_acc
	
	return rel_const_acc



def get_succ_pats(dt,unique_rel_id,direction):
	rel_succ_pats ={}
	for rel_id in unique_rel_id:
		df_sub = dt[(dt['relID']==rel_id) & (dt['patDir']==direction)]
		#print(df_sub)
		unique_pat_id = list(set(df_sub['patID'].to_list()))
		succ_pats = {}
		for patid in unique_pat_id:
			df_sub1 = df_sub[(df_sub['patID']==patid)]
			atleast_one_corr = 0
			lab = df_sub1['v_sub'].to_list()
			pred = df_sub1['pred_v_sub'].to_list()
			scores = df_sub1['scores'].to_list()
			if 1 in scores:
				atleast_one_corr =1

			succ_pats[patid] = atleast_one_corr
		rel_succ_pats[rel_id] = succ_pats
	
	return rel_succ_pats

def get_succ_vsub(dt,unique_rel_id,direction):
	rel_succ_vsub ={}
	for rel_id in unique_rel_id:
		df_sub = dt[(dt['relID']==rel_id) & (dt['patDir']==direction)]
		#print(df_sub)
		unique_entities_pair_id = list(set(df_sub['ent_pair_id'].to_list()))
		succ_vsub = {}
		for ent_pair in unique_entities_pair_id:
			df_sub1 = df_sub[(df_sub['ent_pair_id']==ent_pair)]
			atleast_one_corr = 0
			lab = df_sub1['v_sub'].to_list()
			pred = df_sub1['pred_v_sub'].to_list()
			scores = df_sub1['scores'].to_list()
			if 1 in scores:
				atleast_one_corr =1

			succ_vsub[ent_pair] = atleast_one_corr
		rel_succ_vsub[rel_id] = succ_vsub
	
	return rel_succ_vsub


def get_unk_know_const(dt,unique_rel_id,direction):
	#test_index = [1, 8, 9, 11, 12, 15, 19, 24, 28, 31, 32, 34, 36, 38, 42, 45, 46, 49, 51, 53]
	if isTestRun:
		selected_indexes = test_index
	else:
		selected_indexes = [int(ss) for ss in unique_rel_id] # run for all

	unk_dataframe = pd.DataFrame([])
	know_dataframe = pd.DataFrame([])
	for rel_id in unique_rel_id:
		if int(rel_id) in selected_indexes:
			df_sub = dt[(dt['relID']==rel_id) & (dt['patDir']==direction)]
			#print(df_sub)
			unique_pat_id = list(set(df_sub['patID'].to_list()))
			for patid in unique_pat_id:
				df_sub1 = df_sub[(df_sub['patID']==patid)]
				scores = df_sub1['scores'].to_list()
				if 1 in scores:
					know_dataframe = pd.concat([know_dataframe, df_sub1], ignore_index=True)
				else:
					unk_dataframe = pd.concat([unk_dataframe , df_sub1], ignore_index=True)


	val_unique_rel_id = [x for x in unique_rel_id if int(x) in selected_indexes]
	if len(know_dataframe)!= 0:
		rel_const = get_rel_consistencies(know_dataframe,val_unique_rel_id,direction)
		#print(rel_const)
		avg_rel_const =  [np.array(list(k.values())).mean() for k in list(rel_const.values())]
		avg_const = sum(avg_rel_const)/len(avg_rel_const)
		avg_know_const = avg_const
	else:
		#not defined
		avg_know_const = -1.0

	if len(unk_dataframe)!= 0:
		rel_const = get_rel_consistencies(unk_dataframe,val_unique_rel_id,direction)
		#print(rel_const)
		avg_rel_const =  [np.array(list(k.values())).mean() for k in list(rel_const.values())]
		avg_const = sum(avg_rel_const)/len(avg_rel_const)
		avg_unk_const = avg_const
	else:
		avg_unk_const = -1.0

	return avg_unk_const,avg_know_const



if nature == "strict":
	
	scores = calculate_all_accuracy_scores(dt)
	#print(np.array(scores).mean())

	dt['scores'] = scores

	rel_accuracy = dt.groupby(['relID'])['scores'].mean()
	rel_forw_back_Acc = dt.groupby(['relID','patDir'])['scores'].mean()


	#forward consistency

	unique_rel_id = list(set(dt['relID'].to_list()))
	forward_rel_const = get_rel_consistencies(dt,unique_rel_id,'forward')
	backward_rel_const = get_rel_consistencies(dt,unique_rel_id,'backward')

	avg_forward_rel_const =  [np.array(list(k.values())).mean() for k in list(forward_rel_const.values())]
	avg_backward_rel_const = [np.array(list(k.values())).mean() for k in list(backward_rel_const.values())]

	avg_forward_const = sum(avg_forward_rel_const)/len(avg_forward_rel_const)
	avg_backward_const = sum(avg_backward_rel_const)/len(avg_backward_rel_const)


	forward_const_acc = get_cons_acc(dt,unique_rel_id,'forward')
	backward_const_acc = get_cons_acc(dt,unique_rel_id,'backward')

	avg_forward_rel_const_acc =  [np.array(list(k.values())).mean() for k in list(forward_const_acc.values())]
	avg_backward_rel_const_acc = [np.array(list(k.values())).mean() for k in list(backward_const_acc.values())]

	avg_forward_cons_acc = sum(avg_forward_rel_const_acc)/len(avg_forward_rel_const_acc)
	avg_backward_cons_acc = sum(avg_backward_rel_const_acc)/len(avg_backward_rel_const_acc)


	forward_succ_pats = get_succ_pats(dt,unique_rel_id,'forward')
	backward_succ_pats = get_succ_pats(dt,unique_rel_id,'backward')

	avg_forward_succ_pats =  [np.array(list(k.values())).mean() for k in list(forward_succ_pats.values())]
	avg_backward_succ_pats = [np.array(list(k.values())).mean() for k in list(backward_succ_pats.values())]

	avg_forward_succ_pats = sum(avg_forward_succ_pats)/len(avg_forward_succ_pats)
	avg_backward_succ_pats = sum(avg_backward_succ_pats)/len(avg_backward_succ_pats)

	forward_succ_vsub = get_succ_vsub(dt,unique_rel_id,'forward')
	backward_succ_vsub = get_succ_vsub(dt,unique_rel_id,'backward')

	avg_forward_succ_vsub =  [np.array(list(k.values())).mean() for k in list(forward_succ_vsub.values())]
	avg_backward_succ_vsub = [np.array(list(k.values())).mean() for k in list(backward_succ_vsub.values())]

	avg_forward_succ_vsub = sum(avg_forward_succ_vsub)/len(avg_forward_succ_vsub)
	avg_backward_succ_vsub = sum(avg_backward_succ_vsub)/len(avg_backward_succ_vsub)

	forward_unk_const, forward_know_const = get_unk_know_const(dt,unique_rel_id,'forward')
	backward_unk_const, backward_know_const = get_unk_know_const(dt,unique_rel_id,'backward') #0,0
	

	with open(output_fine_name,"w") as f:
		csvwriter = csv.writer(f)
		csvwriter.writerow(["relation_id","avg_accuracy","forward_accuracy","backward_accuracy",
			                "avg_cons","forward_cons","backward_cons","avg_cons_acc","forward_cons_acc","backward_cons_acc",
			                "avg_succ_pats","forward_succ_pats","backward_succ_pats",
			                "avg_succ_vsub","forward_succ_vsub","backward_succ_vsub"])
		#test_index = [1,8,9,11,12,15,19,24,28,31,32,34,36,38,42,45,46,49,51,53]
		if isTestRun:
			selected_indexes = test_index
		else:
			selected_indexes = [int(ss) for ss in unique_rel_id]

		for re in range(len(unique_rel_id)):
			if int(unique_rel_id[re]) in selected_indexes:
				for_acc = rel_forw_back_Acc[unique_rel_id[re]]['forward']
				bac_acc = rel_forw_back_Acc[unique_rel_id[re]]['backward']
				avg_acc =  (for_acc + bac_acc)/2

				for_cons = np.array(list(forward_rel_const[unique_rel_id[re]].values())).mean()
				bac_cons = np.array(list(backward_rel_const[unique_rel_id[re]].values())).mean()
				avg_cons = (for_cons + bac_cons) /2
			
				for_cons_acc = np.array(list(forward_const_acc[unique_rel_id[re]].values())).mean()
				bac_cons_acc = np.array(list(backward_const_acc[unique_rel_id[re]].values())).mean()
				avg_cons_acc = (for_cons_acc + bac_cons_acc) /2

				for_succ_pat = np.array(list(forward_succ_pats[unique_rel_id[re]].values())).mean()
				bac_succ_pat = np.array(list(backward_succ_pats[unique_rel_id[re]].values())).mean()
				avg_succ_pat = (for_succ_pat + bac_succ_pat) /2

				for_succ_vsub = np.array(list(forward_succ_vsub[unique_rel_id[re]].values())).mean()
				bac_succ_vsub = np.array(list(backward_succ_vsub[unique_rel_id[re]].values())).mean()
				avg_succ_vsub = (for_succ_vsub + bac_succ_vsub) /2

				row = [unique_rel_id[re],avg_acc,for_acc,bac_acc,avg_cons,for_cons,bac_cons,avg_cons_acc,
			        for_cons_acc,bac_cons_acc,avg_succ_pat,for_succ_pat,bac_succ_pat,avg_succ_vsub,for_succ_vsub,
			        bac_succ_vsub]
				csvwriter.writerow(row)

		csvwriter.writerow([])
		csvwriter.writerow(['avg_unk_cons','forward_unk_cons','backward_unk_cons','avg_know_cons','forward_know_cons','backward_know_cons'])
		csvwriter.writerow([(forward_unk_const+backward_unk_const)/2,forward_unk_const,backward_unk_const,(forward_know_const+backward_know_const)/2,forward_know_const,backward_know_const])


	