#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""inference_utils.py: script to implement llms generation"""

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
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoModelForMaskedLM, BitsAndBytesConfig
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fire
from datasets import load_dataset
from dotenv import load_dotenv
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
from transformers.tokenization_utils_base import logger as tokenization_logger

from utils.prompter import Prompter


model = None
tokenizer = None


def load_model(model_name,exptype):
    global model
    global tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name=="BERT_BASE":
        model_name = "/home/models/bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        model.eval()
    elif model_name=="GPT2":
        model_name = "/home/models/gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
    elif model_name=="GPTJ":
        model_name = "EleutherAI/gpt-j-6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
    elif model_name=="GPTNEOX":
        model_name = "EleutherAI/gpt-neox-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
    elif model_name=="ROBERTA_BASE":
        model_name = "/home/models/roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()
    elif model_name=="LLAMA13":
        device_map =  {'': 0}
        tokenizer = AutoTokenizer.from_pretrained("/home/models/llama-13b-hf",unk_token="<unk>",bos_token="<s>",eos_token="</s>")
        
        if exptype == "default":
            model = AutoModelForCausalLM.from_pretrained("/home/models/llama-13b-hf",load_in_8bit=True,torch_dtype=torch.float16, device_map=device_map)
        elif exptype == "finetuned":
            basemodel = '/home/models/baseline_sft_model' # replace with basemodel, could be default model or MTIT SFT model
            adapter = './ift_ctsrlD/' # Replace the adapter, it could be an MTIT adapter combined with default or RL adapter combined withbMTIT base model
            model = AutoModelForCausalLM.from_pretrained(basemodel,load_in_8bit=True,torch_dtype=torch.float16, device_map=device_map)
            model = PeftModel.from_pretrained(model,adapter, torch_dtype=torch.float16, device_map=device_map)

        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        #model = model.to(device)
        model.eval()
    elif model_name=="LLAMA7":
        device_map =  {'': 0}
        tokenizer = AutoTokenizer.from_pretrained("yahma/llama-7b-hf",unk_token="<unk>",bos_token="<s>",eos_token="</s>")
        if exptype == "default":
            model = AutoModelForCausalLM.from_pretrained("yahma/llama-7b-hf",load_in_8bit=True,torch_dtype=torch.float16, device_map=device_map)
        elif exptype == "finetuned":
            basemodel = '/home/models/sft_model_llama7' # replace with basemodel, could be default model or MTIT SFT model
            adapter = './mtit_tsrl_llama7/' # Replace the adapter, it could be an MTIT adapter combined with default or RL adapter combined withbMTIT base model
            model = AutoModelForCausalLM.from_pretrained(basemodel,load_in_8bit=True,torch_dtype=torch.float16, device_map=device_map)
            model = PeftModel.from_pretrained(model,adapter, torch_dtype=torch.float16, device_map=device_map)

        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.eval()
        #model = model.to(device)
    elif model_name=="LLAMA30":
        device_map =  {'': 0}
        tokenizer = AutoTokenizer.from_pretrained("/home/models/llama-30b-hf",unk_token="<unk>",bos_token="<s>",eos_token="</s>")
        if exptype == "default":
            model = AutoModelForCausalLM.from_pretrained("/home/models/llama-30b-hf",load_in_8bit=True,torch_dtype=torch.float16, device_map=device_map)
        elif exptype == "finetuned":
            basemodel = '/home/models/llama-30b-hf' # replace with basemodel, could be default model or MTIT SFT model
            adapter = './lora_llama30_cont_aux5_2/' # Replace the adapter, it could be an MTIT adapter combined with default or RL adapter combined withbMTIT base model
            quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True,load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(basemodel,torch_dtype=torch.float16, device_map=device_map,quantization_config=quantization_config)
            model = PeftModel.from_pretrained(model,adapter, torch_dtype=torch.float16, device_map=device_map)

        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.eval()
    return True

#load_model("LLAMA13")


def get_bad_vocab_list(gold_tokens):
    global tokenizer
    goldlabel = tokenizer.encode(" ".join(gold_tokens))
    bad = [[id] for id in range(tokenizer.vocab_size) if id not in goldlabel]
    return bad

def next_prediction(prompt, num_words, gold_tokens, model_name, bad_list, isopenvocab):
	
    global model
    global tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    goldlabel = tokenizer.encode(" ".join(gold_tokens))
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    #generated_ids = model.generate(input_ids, force_words_ids=[goldlabel],num_beams=2, temperature=0.9, max_new_tokens=9)
    if isopenvocab:
        generated_ids = model.generate(input_ids=input_ids,  repetition_penalty=1.25, pad_token_id=tokenizer.eos_token_id, num_beams=2, temperature=0.9, max_new_tokens=40)
    else:
        generated_ids = model.generate(input_ids , bad_words_ids=bad_list,num_beams=2, temperature=0.9, max_new_tokens=9)
    generated_text = tokenizer.decode(generated_ids[0]) 
    try:
        pred = generated_text.split(prompt)[1]
    except:
        pred =""
    pred = pred.strip().replace("\n"," ")
    return pred

	#goldlabel = tokenizer.encode(" ".join(gold_tokens))
	#force_words_ids


# generation from LLaMA series models
def next_prediction_llama(prompt, num_words, gold_tokens, model_name,  bad_list, isopenvocab):
    global model
    global tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
	
    input_ids = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    #input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if isopenvocab:
        generated_ids = model.generate(input_ids=input_ids,  repetition_penalty=1.25, pad_token_id=tokenizer.eos_token_id, num_beams=2, temperature=0.9, max_new_tokens=50)
    else:
        generated_ids = model.generate(input_ids=input_ids,  bad_words_ids=bad_list, repetition_penalty=1.25, pad_token_id=tokenizer.eos_token_id, num_beams=2, temperature=0.9, max_new_tokens=40)
    
    if len(prompt)>1:
        generated_text = [tokenizer.decode(x) for x in generated_ids]
    else:
        generated_text = [tokenizer.decode(generated_ids[0])]
    #print("next:",generated_text)
    pred=[]
    for k in range(len(generated_text)):
        try:
            pred.append(generated_text[k].split(prompt[k])[1].strip().replace("\n"," "))

        except:
            pred.append("")
    return pred



def mask_prediction(sent, num_words, gold_tokens, model_name):
	global model
	global tokenizer

	return " "


def get_context_dictionary():

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

    return context

