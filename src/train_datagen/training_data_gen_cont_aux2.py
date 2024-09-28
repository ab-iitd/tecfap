
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""training_data_gen_cont_aux2.py: script to generate training data"""

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
from read_temporal_data import *
import random
import json
from copy import deepcopy
projpath = Path(os.path.dirname(os.path.realpath(__file__))).parent.absolute().parent.absolute()

# read TEMP-COFAC dataset
candidates_relations = read_candidates(os.path.join(projpath,'dataset/temp-cofac/candidates'))
entities_relations = read_entities(os.path.join(projpath,'dataset/temp-cofac/samples'))
strict_relations = read_patterns(os.path.join(projpath,'dataset/temp-cofac/strict'))

#print(candidates_relations)

final_data = []

#train_idx = random.sample([k for k in range(66)], 46)
#print(train_idx)

# read train indexes
dftrain = pd.read_csv(os.path.join(projpath,'dataset/temp-cofac/train_index.csv'))
train_idx = dftest['train_index'].tolist()
test_idx =[]
#train_idx = [55, 21, 25, 58, 61, 62, 4, 48, 43, 52, 20, 63, 5, 10, 6, 50, 60, 27, 56, 30, 23, 29, 7, 33, 14, 59, 3, 54, 39, 64, 35, 2, 41, 40, 0, 44, 18, 65, 47, 37, 17, 13, 57, 22, 26, 16]

#choices: IT_BASE, IT_CONT, MTIT_CONT
trdata_setting = "IT_BASE"

# output file path for training data
output_training_file = 'tefcop_tr_cont_aux5_2.json'


instruction = "Complete the given sentence with correct phrase"

context = {0:"here is the list of albums released by linkin park - ",
1:"here is the list of albums released by Euphoria - ",
2:"here is the list of vehicles released by Tata - ",
3:"here is the list of vehicles released by Maruti - ",
4:"here is the list of os released by Apple - ",
5:"here is the list of countries playing test cricket - ",
6:"here is the list of Independent Countries - ",
7:"here is the list of presidents of United States of America - ",
8:"here is the list of presidents of India - ",
9:"here is the list of CEO's of IBM - ",
10:"here is the list of countries joined WTO - ",
11:"here is the list of countries signed NPT - ",
12:"here is the list of countries signed Geneva Protocol - ",
13:"here is the list of Films released by Rajshree Production - ",
14:"here is the list of Films released by Paramount Pictures - ",
15:"here is the list of Films released by Warner Bros. - ",
16:"here is the list of countries joined Arab League - ",
17:"here is the list of countries hosted G20 - ",
18:"here is the list of satellites launched by ISRO - ",
19:"here is the list of satellites launched by NASA - ",
20:"here is the list of satellites launched by ESA - ",
21:"here is the list of Movies directed by Christopher Nolan - ",
22:"here is the list of elements in periodic table - ",
23:"here is the list of books written by John Grisham - ",
24:"here is the list of mughal emperor - ",
25:"here is the list of CEO's of Microsoft - ",
26:"here is the list of CEO's of Apple - ",
27:"here is the list of countries hosted F1 Grand Prix for the 1970 season - ",
28:"here is the list of albums released by Beatles - ",
29:"here is the list of albums released by Kanye West - ",
30:"here is the list of albums released by Eminem - ",
31:"here is the list of android os released by Google - ",
32:"here is the list of cricketers achieved 10000 ODI runs milestone - ",
33:"here is the list of countries joined NATO - ",
34:"here is the list of best picture award at academy awards - ",
35:"here is the list of best feature film award at national awards - ",
36:"here is the list of states joined United States - ",
37:"here is the list of game of the year award at game awards - ",
38:"here is the list of best independent game award at game awards - ",
39:"here is the list of Movies directed by Quentin Tarantino - ",
40:"here is the list of books written by Robin Cook - ",
41:"here is the list of CEO's of Infosys - ",
42:"here is the list of CEO's of Volkswagen - ",
43:"here is the list of best original song award at academy awards - ",
44:"here is the list of countries hosted ICML - ",
45:"here is the list of Movies directed by Yash Chopra - ",
46:"here is the list of Movies directed by S. S. rajmouli - ",
47:"here is the list of satellites launched by ROSCOSMOS - ",
48:"here is the list of books written by Chetan Bhagat - ",
49:"here is the list of Academy Award for Best Cinematography - ",
50:"here is the list of Movies directed by Ridley Scott - ",
51:"here is the list of Movies directed by Francis Ford Coppola - ",
52:"here is the list of countries hosted FIFA U-17 World Cup - ",
53:"here is the list of countries hosted FIFA Futsal World Cup - ",
54:"here is the list of countries hosted F1 Grand Prix for the 2019 season - ",
55:"here is the list of countries hosted motoGP Grand Prix for the 2019 season - ",
56:"here is the list of primeministers of united kingdom - ",
57:"here is the list of academy awards for best costume design - ",
58:"here is the list of alubum of the year awards at grammy award - ",
59:"here is the list of viceroy of india - ",
60:"here is the list of presidents of American Sociological Association - ",
61:"here is the list of presidents of American Psychological Association - ",
62:"here is the list of presidents of American Physiological Association - ",
63:"here is the list of presidents of American Political Science Association - ",
64:"here is the list of presidents of American Philosophical Association - ",
65:"here is the list of presidents of Virginia Tech - "
}

for ind,rel in enumerate(strict_relations):
    if ind in train_idx:
        cand = deepcopy(entities_relations[ind])
        entities = entities_relations[ind]
        for pat_ind, pat in enumerate(rel):
            patrn = pat['pattern'].lower()
            pat_direction = pat['direction']
            q_sub =""
            v_sub =""
            sent =""
            if(pat_direction=='backward'):
                for i in range(1,len(entities)):
                    q_sub = entities[i]
                    v_sub = entities[i-1]
                    sent = patrn.replace('[x]',q_sub)
                    sent = sent.replace(' [y]',"")
                    random.shuffle(cand)
                    if trdata_setting == "IT_BASE":
                        input_text = sent
                    elif trdata_setting == "IT_CONT" or trdata_setting == "MTIT_CONT":
                        input_text = context[ind] + ", ".join(cand)+". "+ sent
                    #output_text = v_sub
                    #output_text = input_text.strip() + " "+ v_sub
                    output_text = sent.strip() + " "+ v_sub
                    final_data.append({"instruction":instruction,"input":input_text,"output":output_text})
            if(pat_direction=='forward'):
                for i in range(len(entities)-1):
                    q_sub= entities[i]
                    v_sub = entities[i+1]
                    sent = patrn.replace('[x]',q_sub)
                    sent = sent.replace(' [y]',"")
                    random.shuffle(cand)
                    if trdata_setting == "IT_BASE":
                        input_text = sent
                    elif trdata_setting == "IT_CONT" or trdata_setting == "MTIT_CONT":
                        input_text = context[ind] + ", ".join(cand)+". "+ sent
                    #output_text = v_sub
                    #output_text = input_text.strip() + " "+ v_sub
                    output_text = sent.strip() + " "+ v_sub
                    final_data.append({"instruction":instruction,"input":input_text,"output":output_text})       

    else:
        test_idx.append(ind)


if trdata_setting == "MTIT_CONT":
    aux_instruct_5 = "Predict if the given sentences are paraphrased or similar in context"
    for ind,rel in enumerate(strict_relations):
        if ind in train_idx:
            train_idx_2 = deepcopy(train_idx)
            train_idx_2.pop(train_idx_2.index(ind))
            #cand = deepcopy(entities_relations[ind])

            for d in range(4):#3
                a1 = random.sample([k for k in range(16)], 1)[0]
                if a1 <8:
                    a2 = random.sample([k for k in range(8) if k != a1], 1)[0]
                else:
                    a2 = random.sample([k for k in range(8,16) if k != a1], 1)[0]

                p1 = rel[a1]['pattern'].lower()
                p2 = rel[a2]['pattern'].lower()


                if a1 < 8:
                    e = random.sample([k for k in range(1,len(entities_relations[ind]))], 1)[0]
                    s1 = p1.replace('[x]',entities_relations[ind][e])
                    s2 = p2.replace('[x]',entities_relations[ind][e])
                    s1 = s1.replace(' [y]',"")
                    s2 = s2.replace(' [y]',"")
                    o1 = s1 + " " + entities_relations[ind][e-1]
                    o2 = s2 + " " + entities_relations[ind][e-1]

                else:
                    e = random.sample([k for k in range(0,len(entities_relations[ind])-1)], 1)[0]
                    s1 = p1.replace('[x]',entities_relations[ind][e])
                    s2 = p2.replace('[x]',entities_relations[ind][e])
                    s1 = s1.replace(' [y]',"")
                    s2 = s2.replace(' [y]',"")
                    o1 = s1 + " " + entities_relations[ind][e+1]
                    o2 = s2 + " " + entities_relations[ind][e+1]


                final_data.append({"instruction":aux_instruct_5,"input":"sentence 1: "+o1 + " \n sentence 2: "+ o2,"output": "True"})

            for d in range(4):#3
                a1 = random.sample([k for k in range(16)], 1)[0]
                
                if d<2:#==0
                    if a1 <8:
                        a2 = random.sample([k for k in range(8,16) if k != a1], 1)[0]
                    else:
                        a2 = random.sample([k for k in range(8) if k != a1], 1)[0]

                    p1 = rel[a1]['pattern'].lower()
                    p2 = rel[a2]['pattern'].lower()

                    if a1 < 8:
                        e = random.sample([k for k in range(1,len(entities_relations[ind]))], 1)[0]
                        s1 = p1.replace('[x]',entities_relations[ind][e])
                        s1 = s1.replace(' [y]',"")
                        o1 = s1 + " " + entities_relations[ind][e-1]
                    else:
                        e = random.sample([k for k in range(0,len(entities_relations[ind])-1)], 1)[0]
                        s1 = p1.replace('[x]',entities_relations[ind][e])
                        s1 = s1.replace(' [y]',"")
                        o1 = s1 + " " + entities_relations[ind][e+1]

                    if a2 < 8:
                        e = random.sample([k for k in range(1,len(entities_relations[ind]))], 1)[0]
                        s2 = p2.replace('[x]',entities_relations[ind][e])
                        s2 = s2.replace(' [y]',"")
                        o2 = s2 + " " + entities_relations[ind][e-1]
                    else:
                        e = random.sample([k for k in range(0,len(entities_relations[ind])-1)], 1)[0]
                        s2 = p2.replace('[x]',entities_relations[ind][e])
                        s2 = s2.replace(' [y]',"")
                        o2 = s2 + " " + entities_relations[ind][e+1]
                else:
                    p1 = rel[a1]['pattern'].lower()

                    if a1 < 8:
                        e = random.sample([k for k in range(1,len(entities_relations[ind]))], 1)[0]
                        s1 = p1.replace('[x]',entities_relations[ind][e])
                        s1 = s1.replace(' [y]',"")
                        o1 = s1 + " " + entities_relations[ind][e-1]
                    else:
                        e = random.sample([k for k in range(0,len(entities_relations[ind])-1)], 1)[0]
                        s1 = p1.replace('[x]',entities_relations[ind][e])
                        s1 = s1.replace(' [y]',"")
                        o1 = s1 + " " + entities_relations[ind][e+1]


                    random_rel = random.sample(train_idx_2, 1)[0]
                    p_idx = random.sample([k for k in range(16)], 1)[0]
                    p2 = strict_relations[random_rel][p_idx]['pattern'].lower()

                    if p_idx < 8:
                        e = random.sample([k for k in range(1,len(entities_relations[random_rel]))], 1)[0]
                        s2 = p2.replace('[x]',entities_relations[random_rel][e])
                        s2 = s2.replace(' [y]',"")
                        o2 = s2 + " " + entities_relations[random_rel][e-1]
                    else:
                        e = random.sample([k for k in range(0,len(entities_relations[random_rel])-1)], 1)[0]
                        s2 = p2.replace('[x]',entities_relations[random_rel][e])
                        s2 = s2.replace(' [y]',"")
                        o2 = s2 + " " + entities_relations[random_rel][e+1]
                final_data.append({"instruction":aux_instruct_5,"input":"sentence 1: "+o1 + " \n sentence 2: "+ o2,"output": "False"})

print(test_idx)

f = open(output_training_file, 'w', encoding='utf-8')
json.dump(final_data, f, ensure_ascii=False, indent=4)
f.close()

