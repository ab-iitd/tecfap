#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""infernece.py: main script to run test inferences with TEMP-COFAC dataset"""

__author__ = "Ashutosh Bajpai"
__copyright__ = ""
__license__ = ""
__project__= "TeCFaP"
__version__ = "1.0.0"
__maintainer__ = "Ashutosh Bajpai"

#_______________________________________________________________________________________________________________

import sys,os
os.environ['CURL_CA_BUNDLE'] = ''
import pandas as pd
import json
import csv
from copy import deepcopy
from read_temporal_data import *
#from mask_next_prediction import *
#from inference_utils import *
import random
from pathlib import Path

projpath = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute().parent.absolute()

model_type = "decoder"  # "encoder for MLM, "decoder "Generative
model_name = "LLAMA13"  # BERT_BASE, GPT2, GPTJ, GPTNEOX, ROBERTA_BASE, LLAMA13, LLAMA7
shots = 0 
output_filename = "strict_"+model_name+".csv"
exptype = 'default' # 'default' to run default LLM and 'finetune' to run fine-tuned LLM 
isTestRun = True # true to run inferences on only test set, false to run on all
isContext = True # true for with context setting and false for without context setting


# read TEMP-COFAC dataset
candidates_relations = read_candidates(os.path.join(projpath,'dataset/temp-cofac/candidates'))
entities_relations = read_entities(os.path.join(projpath,'dataset/temp-cofac/samples'))
strict_relations = read_patterns(os.path.join(projpath,'dataset/temp-cofac/strict'))


if isTestRun:
    # reading test indexes
    dftest = pd.read_csv(os.path.join(projpath,'dataset/temp-cofac/test_index.csv'))
    test_index = dftest['test_index'].tolist()
    #test_index =[1, 8, 9, 11, 12, 15, 19, 24, 28, 31, 32, 34, 36, 38, 42, 45, 46, 49, 51, 53] 

    strict_relations = [strict_relations[i] for i in test_index]
    entities_relations = [entities_relations[i] for i in test_index]
    candidates_relations=[candidates_relations[i] for i in test_index]

# load model
load_model(model_name, exptype)

outfile = open(output_filename, 'w')
csvwriter = csv.writer(outfile)
csvwriter.writerow(["modelName","relID","patID","pat","patDir","patNat","ent_pair_id","q_sub","v_sub","pred_v_sub"]) 

# get context template for each relations
context = get_context_dictionary()

input_sent_list=[]
input_prompts_list = []
output_list =[]
prompter = Prompter("")
#instruction = "given an input sentence, you need to correctly generate following words such that the entity present in input text and gnerated text are temporally conntected through a context present in input sentence"
instruction = "Complete the given sentence with correct phrase"
for ind,rel in enumerate(strict_relations):
    entities = entities_relations[ind]
    #bwd,fwd = get_examples(shots,rel,ind)
    all_patrn = ''
    bad_vocab_list = []#get_bad_vocab_list(candidates_relations[ind])
    cand = deepcopy(entities_relations[ind])
    random.shuffle(cand)
    for pat_ind, pat in enumerate(rel):
        random.shuffle(cand)
        patrn = pat['pattern'].lower()
        pat_id = pat['id']
        pat_direction = pat['direction']
        pat_nat = pat['temporal_nature']
        if(pat_direction=='backward'):
            for i in range(1,len(entities)):
                random.shuffle(cand)
                q_sub = entities[i]
                v_sub = entities[i-1]
                ent_pair_id = str(i)+"_"+str(i-1)
                sent = patrn.replace('[x]',q_sub)
                sent = sent.replace(' [y]',"")
                num_words = 0
                for k in v_sub.split():
                    num_words += 1
                if isContext:
                    ip_text = context[ind] + ", ".join(cand)+". "+ sent
                else:
                    ip_text = sent
                iptext = prompter.generate_prompt(instruction,ip_text) 
                #sent = get_prompt(shots, True, sent, bwd)
                #print(sent)
                input_prompts_list.append(iptext)
                input_sent_list.append(sent)
                
                output_list.append([model_name, ind, pat_id, pat['pattern'].lower(), pat_direction, pat_nat, ent_pair_id,q_sub, v_sub])
                #csvwriter.writerow([model_name, ind, pat_id, pat['pattern'].lower(), pat_direction, pat_nat, ent_pair_id,q_sub, v_sub, pred_v_sub])
        
        if(pat_direction=='forward'):
            for i in range(len(entities)-1):
                random.shuffle(cand)
                q_sub= entities[i]
                v_sub = entities[i+1]
                sent = patrn.replace('[x]',q_sub)
                ent_pair_id = str(i)+"_"+str(i+1)
                sent = sent.replace(' [y]',"")
                num_words = 0
                for k in v_sub.split():
                    num_words += 1
                if isContext:
                    ip_text = context[ind] + ", ".join(cand)+". "+ sent
                else:
                    ip_text = sent
                iptext = prompter.generate_prompt(instruction,ip_text)
                #sent = get_prompt(shots, True, sent, fwd)
                input_sent_list.append(sent)
                input_prompts_list.append(iptext) 
                
                output_list.append([model_name, ind, pat_id, pat['pattern'].lower(), pat_direction, pat_nat, ent_pair_id,q_sub, v_sub])
                #csvwriter.writerow([model_name, ind, pat_id, pat['pattern'].lower(), pat_direction, pat_nat, ent_pair_id, q_sub, v_sub, pred_v_sub])


# run batches to generate the responses
xx =0
out_pred =[]
batch_size = 6
while xx<len(input_prompts_list):
    bch = input_prompts_list[xx:xx+batch_size]
    sen = input_sent_list[xx:xx+batch_size]
    pred_v_subs = next_prediction_llama(bch, None, None, None, None, True)
    #pred_v_subs = bch
    for yy in range(len(bch)):
        if sen[yy] in pred_v_subs[yy]:
            out_pred.append(pred_v_subs[yy].split(sen[yy])[1])
        else:
            out_pred.append(pred_v_subs[yy])
    if (xx%100):
        print(xx)
    xx = xx+batch_size

for xx in range(len(out_pred)):
    output_list[xx].append(out_pred[xx])
    csvwriter.writerow(output_list[xx])

outfile.close()


