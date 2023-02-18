from datasets import load_dataset
import numpy as np 
import pandas as pd 
import re
from transformers import AutoTokenizer, AutoConfig,AutoModelForCausalLM
from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling
import argparse
import os


BASE_GPT2_MODEL="flax-community/gpt-2-spanish"
HUGGINGFACE_PATH_DATASET="jhonparra18/petro-tweets"
N_TOKENS_CONTEXT = 128 #number of tokens for context to create next word
os.environ['WANDB_PROJECT']="gpt2-text-generation"

tokenizer=AutoTokenizer.from_pretrained(BASE_GPT2_MODEL)
config = AutoConfig.from_pretrained(
    BASE_GPT2_MODEL,
    vocab_size=len(tokenizer),
    n_ctx=N_TOKENS_CONTEXT,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


def clean_text(instance):
    instance['text']=[txt.replace("\n"," ").replace("  "," ") for txt in instance['Tweet']]
    instance['text'] =[re.sub(r"http\S+", "",txt).strip() for txt in instance['text']]
    instance['text']
    return instance

def tokenize(instance):
    outputs = tokenizer(
        instance["text"],
        truncation=True,
        max_length=N_TOKENS_CONTEXT
    )
    return outputs


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Twitter AI assistant")
    parser.add_argument("-tr", "--train_prop",type=float,default=0.9,help="train proportion of data", dest="train_prop")
    parser.add_argument("-e", "--epochs",type=int,default=30, help="Number of Epochs", dest="n_epochs")
    parser.add_argument("-bs", "--batch_size",type=int,default=20, help="Batch Size", dest="batch_size")
    parser.add_argument("-n", "--runname",type=str,default="twitter-ai-assistant", help="Run name for training-wandb runname and hf repo", dest="run_name")
    parser.add_argument("-u","--user",type=str,default="jhonparra18", help="HuggingFace User name", dest="user_name")
    args = parser.parse_args()

    REPO_NAME=f"{args['user_name']}/{args['run_name']}"

    ## change source if your data 
    dataset = load_dataset(HUGGINGFACE_PATH_DATASET,split="train") 

    ##make sure tweets have at least 3 tokens
    dataset=dataset.map(clean_text,batched=True).remove_columns(['Date','User','Tweet'])\
            .filter(lambda instance: len(instance['text'].split(" "))>3)\
            .shuffle(seed=444).train_test_split(test_size=0.1)

    data_tokenized = dataset.map(
        tokenize, batched=True,remove_columns="text")


    tokenizer.pad_token = tokenizer.eos_token 
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) ##causal language modeling (mlm=False)
    model = AutoModelForCausalLM.from_pretrained(BASE_GPT2_MODEL,config=config)


    args = TrainingArguments(
    output_dir=REPO_NAME,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=1000,
    gradient_accumulation_steps=2,
    num_train_epochs=30,
    weight_decay=0.1,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    save_steps=1000,
    fp16=True,
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to="wandb",
    run_name=args['run_name']
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=data_tokenized["train"],
        eval_dataset=data_tokenized["test"],
    )

    trainer.train()
    trainer.push_to_hub(commit_message="training checkpoint updated")
