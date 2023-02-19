from train import N_TOKENS_CONTEXT
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


tokenizer = AutoTokenizer.from_pretrained("jhonparra18/petro-twitter-assistant-30ep")

model = AutoModelForCausalLM.from_pretrained("jhonparra18/petro-twitter-assistant-30ep")


torch.manual_seed(444)    ##for reproducibility
tokenizer.padding_side="left" ##start padding from left to right
tokenizer.pad_token = tokenizer.eos_token 


def text_completion(input_text:str,max_len:int=100):

  input_ids = tokenizer([input_text], return_tensors="pt",truncation=True,max_length=128)
  outputs = model.generate(**input_ids, do_sample=True, max_length=max_len,top_k=100,top_p=0.95)
  out_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  return out_text


if __name__=="__main__":

    print(text_completion("este gobierno no es corrupto, solo gobernamos con ")) ## desidios por su propio gob, pero no gobernamos por el hambre. No permitamos que el hambre impacte en la humanidad. Vamos por la lucha de clases, vamos por el socialismo. Viva y Viva la Colombia Humana. #YoVotoPetroPresidente @JuanManSantosD.G. @CamiloRomero @FinalPrecursed la Colombia Humana en