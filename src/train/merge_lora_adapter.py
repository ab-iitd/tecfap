#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""merge_lora_adapter.py: script to merge lora fine-tuned adaper with base model"""

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
from transformers import AutoModelForMaskedLM
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
import time

model = None
tokenizer = None
device_map ="auto"

base_model = '/home/models/llama-13b-hf' #replace with base model
lora_adapter = './lora_llama30_cont_aux5_2/' # replace with the lora adapter checkpoint

output_model  = 'sft_llama13/' # output merged model

tokenizer = AutoTokenizer.from_pretrained(base_model,unk_token="<unk>",bos_token="<s>",eos_token="</s>")
model = AutoModelForCausalLM.from_pretrained(base_model,torch_dtype=torch.float16, device_map=device_map)

model = PeftModel.from_pretrained(model,lora_adapter, torch_dtype=torch.float16, device_map=device_map)

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.eval()

model = model.merge_and_unload()

model.save_pretrained(output_model)
tokenizer.save_pretrained(output_model)
