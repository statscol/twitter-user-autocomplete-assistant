from train import N_TOKENS_CONTEXT
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


tokenizer = AutoTokenizer.from_pretrained("jhonparra18/petro-twitter-assistant")

model = AutoModelForCausalLM.from_pretrained("jhonparra18/petro-twitter-assistant")


torch.manual_seed(444)    ##for reproducibility
tokenizer.padding_side="left" ##start padding from left to right
tokenizer.pad_token = tokenizer.eos_token 


def text_completion(input_text:str,max_len:int=100):

  input_ids = tokenizer([input_text], return_tensors="pt",truncation=True,max_length=128)
  outputs = model.generate(**input_ids, do_sample=True, max_length=max_len)
  out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  return out_text


if __name__=="__main__":

    print(text_completion("este gobierno no es corrupto, solo gobernamos con ")) ##los que nos pusieron votos así sean de dudosa reputación