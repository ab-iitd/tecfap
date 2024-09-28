
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""rl_ppo.py: main script for CTSRL finetuning"""
"""credit: code inspired from https://github.com/CarperAI/trlx/blob/main/examples/ppo_sentiments.py """

__author__ = "Ashutosh Bajpai"
__copyright__ = ""
__license__ = ""
__project__= "TeCFaP"
__version__ = "1.0.0"
__maintainer__ = "Ashutosh Bajpai"

#_______________________________________________________________________________________________________________

import json
import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
import sys
from typing import List

import torch
from datasets import load_dataset
from transformers import pipeline
torch.cuda.empty_cache()
import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import gc 
import os
os.environ['CURL_CA_BUNDLE']=''
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fire
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft.utils.config import TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)
from transformers.tokenization_utils_base import logger as tokenization_logger

from utils.prompter import Prompter

training_data_json_file = "/home/lora/tefcop_tr_cont.json" # replace with training datafile
output_lora_adapter = "/home/lora/mtit_tsrl_llama7/"
# file contains all entities in temporal sseuence for a query, used in CTSRL Smooth
all_entities_4_a_query_file = 'seq_rl_input.csv'
sft_model = "./sft_model_llama7" # replace with the finetuned MTIT or IT model

isCTSRLSmooth = False # true to run CTSRL smooth reward

#play with reward values
cons_reward = 1
fact_reward = 1
# reproduce only for discrete with isCTSRLSmooth = False
alpha = 0.5 # 0.5, 0.66, 0.75
if alpha == 0.5:
    cons_reward = 1
    fact_reward = 1
if alpha == 0.66:
    cons_reward = 2
    fact_reward = 1
if alpha == 0.75:
    cons_reward = 3
    fact_reward = 1


class TokenizerHelper:
    def __init__(
        self, prompter, tokenizer, train_on_inputs, cutoff_len, add_eos_token=True
    ):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token
        self.cutoff_len = cutoff_len

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            # Set padding to 'max_length' instead of False for GPTNeoXTokenizerFast???
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
            and self.add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        return result

    def generate_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
        )
        out_prompt = data_point["output"]
        return {"prms":full_prompt, "label":out_prompt}

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["input_ids"][
                user_prompt_len:
            ]  # could be sped up, probably
        else:
            tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"]

        return tokenized_full_prompt

def load_data():
    """TODO: Not working yet.

    Args:
        config (TrainConfig): _description_

    Returns:
        Tuple: _description_
    """
    # Load the dataset
    dataset = load_dataset("tefcop_tr_cont.json")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, mlm=False)

    # Split the dataset into train, validation and (optionally) test sets
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]

    if "test" in tokenized_dataset:
        test_dataset = tokenized_dataset["test"]
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset, data_collator


#def get_ordered_entities(ent_list):

def llama_config():
    device_map =  {'': 0}
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    return TRLConfig(
        train=TrainConfig(
            seq_length=256,
            epochs=15,
            total_steps=2600, #1280 10
            batch_size=4,
            checkpoint_interval=10000,
            #val_set_size= 1,
            eval_interval=100000,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=False,
            tracker="tensorboard",
        ),
        
        #baseline_sft is an it model sft is an mtit model
        model=ModelConfig(model_path=sft_model, num_layers_unfrozen=0),
        tokenizer=TokenizerConfig(tokenizer_path=sft_model, truncation_side="right"),

        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=12,
            ppo_epochs=4,
            init_kl_coef=0.001,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=50,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(llama_config().to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

        # Just insert your peft config here (the type must be an instance of peft.PeftConfig or a dict).
    config.model.peft_config = LoraConfig(
        r=8,
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.1,
    )  
 
    prompter = Prompter("alpaca")

    
    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str], tokenizer:config.tokenizer):
        #config.model.config.pad_token_id = config.tokenizer.pad_token_id = 0
        #config.model.config.bos_token_id = 1
        #config.model.config.eos_token_id = 2
        #config.model.config.unk_token_id = 0
        #print(prompts)
        original_labels=[]
        predicted_labels=[]
        task_indicator = []
        ent_seq =[]
        tasks = ["Predict if the given sentences are paraphrased or similar in context","Complete the given sentence with correct phrase"]
        for pp in range(len(prompts)):
            #print(prompts[pp])
            #print(outputs[pp])
            if tasks[0] in prompts[pp]:
                task_indicator.append(0)
                original_labels.append(prompts_labels_entity[prompts[pp]].strip())
                predicted_labels.append(outputs[pp].split()[0].strip())
                ent_seq.append('None')
            elif tasks[1] in prompts[pp]:
                task_indicator.append(1)
                if prompts[pp] in prompts_key:
                    original_labels.append(prompts_labels_entity[prompts[pp]].strip())
                    ipsub = prompts_label_input_sub[prompts[pp]]
                    ent_seq.append(prompts_labels_all_ent[prompts[pp]])
                else:
                    key = [i for i in prompts_key if prompts[pp] in i][0]
                    original_labels.append(prompts_labels_entity[key].strip())
                    ent_seq.append(prompts_labels_all_ent[key])
                    ipsub = "None"
            
                if ipsub != "None":
                    try:
                        predicted_labels.append(' '.join(outputs[pp].split(ipsub)[1].split()[:len(original_labels[pp].split())]).strip())
                    except:
                        predicted_labels.append('None')
                else:
                    predicted_labels.append('None')

        scores =[]

        for kk in range(len(original_labels)):
            if original_labels[kk]==predicted_labels[kk] and task_indicator[kk]==0:
                scores.append(cons_reward)#
            elif original_labels[kk]==predicted_labels[kk]:
                scores.append(fact_reward)
            else:
                if isCTSRLSmooth:
                    rw=0
                    if predicted_labels[kk] in ent_seq[kk]:
                        o_idx = ent_seq[kk].index(original_labels[kk])
                        p_idx = ent_seq[kk].index(predicted_labels[kk])
                        if o_idx==0 or o_idx==len(ent_seq[kk])-1:
                            rw = ((len(ent_seq[kk]) - abs(o_idx-p_idx))*1.0)/len(ent_seq[kk])
                        else:
                            if o_idx>p_idx:
                                rw =(((o_idx+1) - abs(o_idx-p_idx))*1.0)/(o_idx+1)
                            else:
                                rw = (((len(ent_seq[kk])-o_idx) - abs(o_idx-p_idx))*1.0)/(len(ent_seq[kk])-o_idx)
                    scores.append(rw)
                else:
                    scores.append(0)
        #scores = [1 if original_labels[kk]==predicted_labels[kk] else 0 for kk in range(len(original_labels))]
        #scores = [len(set(original.split()).intersection(output.split()))/len(set(original.split()).union(output.split())) for (original, output) in zip(original_labels, outputs)]
        #gc.collect()
        return scores

    
    data = load_dataset("json", data_files=training_data_json_file)
    tokenizer_helper = TokenizerHelper(
        prompter, None, None, None, None
    )
    print(data["train"].shuffle())
    train_data = (
             data["train"].shuffle().map(tokenizer_helper.generate_prompt)
        )
    
    prompts = train_data["prms"]

    prompts_out = train_data["label"]
    prompts_labels ={}
    prompts_labels_entity ={}
    prompts_labels_all_entities ={}
    prompts_label_input_sub = {}
    prompts_labels_all_ent={}

    import csv
    data_seq_dict={}
    
    with open(all_entities_4_a_query_file) as csvfile:
        preader = csv.reader(csvfile, delimiter=',')
        for row in preader:
            data_seq_dict[row[0]] = row[1].split(",") 

    for p in range(len(prompts)):
        #input_prm = prompts[p].split("Input:")[-1].split("###")[0].split(".")[-1].strip()
        input_sub = prompts[p].split("Input:")[-1].split("###")[0][-15:].strip()
        prompts_label_input_sub[prompts[p]] = input_sub
        prompts_labels[prompts[p]] = prompts_out[p]
        prompts_labels_entity[prompts[p]] = prompts_out[p].split(input_sub)[-1].strip()
        try:
            prompts_labels_all_ent[prompts[p]] = data_seq_dict[prompts_out[p].strip()]
        except:
            continue
        #prompts_labels_all_entities[prompts[p]] = prompts[p].split("Input:")[-1].split("###")[0].split(".")[0].split("-")[-1].strip()
    prompts_key = prompts_labels.keys()
        #print(prompts[p])
        #print(input_prm)
        #print(prompts_out[p].split(input_prm)[-1].strip())
        #print(prompts[p].split("Input:")[-1].split("###")[0].split(".")[0].split("-")[-1].strip())
    #prompts_labels = train_data["label"]

    #torch.cuda.empty_cache()
    
    trainer_tr = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=None,
        config=config,
    )

    trainer_tr.model.save_pretrained(output_lora_adapter)



if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
