from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
import re
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from trl import SFTTrainer
import datasets
import bitsandbytes as bnb
import huggingface_hub
from statistics import mean
from semscore import ModelPredictionGenerator, EmbeddingModelWrapper



def find_all_linear_names(model): # copied from https://github.com/mzbac/llama2-fine-tune/blob/master/utils.py
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

def print_trainable_parameters(model): # copied from https://github.com/mzbac/llama2-fine-tune/blob/master/utils.py
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

def Get_DesiredOutput(dataframe, prompt, task, tokenizer):
    full_explanations = []

    # get an example for each unique label

    for i, row in dataframe.iterrows():
        lable_samples = pd.DataFrame()
        sample = dataframe.sample(1)
        lable_samples = lable_samples._append(sample)

        # randomize order of lable_samples
        lable_samples = lable_samples.sample(frac=1).reset_index(drop=True)

        chat = []
        
        systemprompt =" You are an expert trained in identifying irony and sarcasm in social media text and explaining the underlying reasoning."
        systemprompt += " Your task is to explain why tweets should be considered ironic.\n\
            Make sure to base your explanation on background knowledge that is not present in the text itself.\n\
            This background knowledge can include common assumptions, factual knowledge and social conventions.\n"

        # append the examples for each unique label in the prompt template
        for samplenr, sample in lable_samples.iterrows():
            example_input = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , sample['text'])\
            .replace("{PLACEHOLDER_FOR_LABEL}", sample['labels'])
            user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', example_input, re.DOTALL)
            system_text = re.search(r'\\begin\[assistant\](.*?)\\end\[assistant\]', example_input, re.DOTALL)
            if samplenr == 0:
                chat.append({"role": "user", "content": systemprompt + user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ").capitalize()})
            else:
                chat.append({"role": "user", "content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ")})
            chat.append({"role": "assistant", "content": system_text.group(0).replace(r"\begin[assistant]", "").replace(r"\end[assistant]", "").replace("  ", " ").capitalize()},)

        
        # get the actual to classify text
        desired_output = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , row['text'])\
        .replace("{PLACEHOLDER_FOR_LABEL}", row['labels'])

        # get text between \begin[user] and \end[user]
        user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', desired_output, re.DOTALL)
        # get text for system , this part is ignored for testing/inference only used as gold for training
        system_text = re.search(r'\\begin\[assistant\](.*?)\\end\[assistant\]', desired_output, re.DOTALL)
        
       
        chat.append({"role": "user", "content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ").capitalize()})
        chat.append({"role": "assistant", "content": system_text.group(0).replace(r"\begin[assistant]", "").replace(r"\end[assistant]", "").replace("  ", " ").capitalize()},)
        desired_output = tokenizer.apply_chat_template(chat, tokenize=False)
        #print(desired_output)
        full_explanations.append(desired_output)
    dataframe["desired_output"] = full_explanations
    return dataframe

def main():

    language = "EN"
    task = "explainer"
    access_token = "hf_placeholder"
    huggingface_hub.login(token=access_token)
    model_name =  "meta-llama/Meta-Llama-3-70B-Instruct"
    save_as_name = model_name.split("/")[1]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, token=access_token )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    prompt = r"\begin[user]Explain why this tweet is ironic:\n\n### Text: {PLACEHOLDER_FOR_INPUTTEXT}\end[user]\begin[assistant]### Explanation:{PLACEHOLDER_FOR_LABEL}\end[assistant]"
    train_data = pd.read_csv(f'data/{language}/train.csv')

    # if the label is a string, lowercase the label
    def check_punctuation(text):
        if text[-1] not in [".", "?", "!"]:
            text += "."
        return text.capitalize()
    
    train_data["labels"] = train_data["explanation"].apply(lambda x: check_punctuation(x))
        
    train_data = Get_DesiredOutput(dataframe=train_data, prompt=prompt, tokenizer=tokenizer, task=task)
    trainset = datasets.Dataset.from_pandas(train_data)

    print(train_data["desired_output"].to_list()[0])

    # Set up PEFT LoRA for fine-tuning.
    lora_config = LoraConfig(
        lora_alpha=16,
        r=128,
        target_modules=find_all_linear_names(base_model),
        task_type="CAUSAL_LM",
    )
    #max_seq_length = 1024

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=trainset,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # This is actually the global batch size for SPMD.
            num_train_epochs=10,
            output_dir=f"./output_{save_as_name}_{task}_{language}",
            eval_accumulation_steps=10,
            dataloader_drop_last = True,  # Required for SPMD.
            hub_private_repo=True,
        ),
        peft_config=lora_config,
        dataset_text_field="desired_output"
    )
    
    trainer.train()
    # save
    trainer.save_model(f"finetuned_models/{save_as_name}_{task}_{language}")
    trainer.push_to_hub(token=access_token)

if __name__ == "__main__":
    main()
    print("Complete")