#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""read_temporal_data.py: script to read TEMP-COFAC dataset"""

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

#reading patterns files
def read_patterns(dirpath):
	pat_relations = []
	for ind in range(66):
		f_path = os.path.join(dirpath,"sub_rel_"+str(ind)+".json")
		f = open(f_path, 'r')
		pats = json.load(f)
		pat_relations.append(pats)
		f.close()
	return pat_relations

#reading entity files
def read_entities(dirpath):
	entities_relations = []
	for ind in range(66):
		f_path = os.path.join(dirpath,"sub_rel_"+str(ind)+".json")
		f = open(f_path, 'r')
		ent = json.load(f)
		entities_relations.append([t['sub_label'].rstrip().lower() for t in ent])
		f.close()
	return entities_relations

#reading candidate files
def read_candidates(dirpath):
	candidates_relations = []
	for ind in range(66):
		f_path = os.path.join(dirpath,"sub_rel_"+str(ind)+".json")
		f = open(f_path, 'r')
		cand = json.load(f)
		candidates_relations.append(cand['candidates'].rstrip().lower().split())
		f.close()
	return candidates_relations


#print(read_candidates("temporal/full_data/candidates"))
#print(read_entities("temporal/full_data/samples"))
#print(read_patterns("temporal/full_data/strict"))
