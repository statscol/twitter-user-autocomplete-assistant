### Twitter User autocomplete-text AI assistant


Use a public twitter account to create an AI assistant for text generation.


- This repo uses a scraping package called `snscrape` to get historical tweets from a user, see `get_data.py`.

- HuggingFace GTP2ForCausalLM is used, the base model and tokenizer for spanish language was taken from HF-HUB user **flax-community/gpt-2-spanish**

## Usage


Create a virtual environment.

```bash
conda create --name <YOUR_ENV_NAME> python=3.9 -y
conda activate <YOUR_ENV_NAME>
pip install -r requirements.txt
```

### Find tweets and save

Get data from your user of preference, modify `get_data.py` to get tweets from your user of preference. (Note: this requires you to set [huggingface credentials](https://huggingface.co/welcome), or you might need to modify the training script to use a csv instead of a HF dataset, `huggingface-cli login`)

```bash
python3 get_data.py
```

## Train using HF Trainer

### set-up Wandb credentials

Set your Weights & Biases access token (Or create an account [here](https://wandb.ai/))

```bash
wandb login
```

### Run the training script

```bash
python3 train.py -tr <YOUR_TRAIN_PARTITION> -e <NUMBER OF EPOCHS> -bs <BATCH_SIZE_TRAIN AND> -n <YOUR_EXPERIMENT_NAME> -u <YOUR_HUGGINGFACE_USERNAME>
```

This will upload the model to HuggingFace Hub, if you want to disable just set the TrainingArguments.push_to_hub to `False`


### Inference 

An inference example can be found in inference.py, however check HuggingFace pipelines ([TextGenerationPipeline](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline))

```bash
python3 inference.py
```


### Demo

See this HuggingFace [Space](https://huggingface.co/spaces/jhonparra18/ColombianPoliticianGPT2TextGeneration).

## To-Do
- Create a torch-lightning script (WIP)
- Testing Encoder-Decoder Approach