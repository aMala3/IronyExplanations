from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import pandas as pd
import re
from peft import  prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import ast
import bitsandbytes as bnb
import huggingface_hub
from transformers.pipelines.pt_utils import KeyDataset
import numpy as np

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


def Set_Prompt_Template(dataframe, example_df, prompt, tokenizer, train=True):
    full_explanations = []

    # get an example for each unique label

    for i, row in dataframe.iterrows():
        lable_samples = pd.DataFrame()
        sample = example_df.sample(1)
        lable_samples = lable_samples._append(sample)

        # randomize order of lable_samples
        lable_samples = lable_samples.sample(frac=1).reset_index(drop=True)

        chat = []
        
        systemprompt =" You are an expert trained in identifying irony and sarcasm in social media text and explaining the underlying reasoning."
        systemprompt += " For each text, we have an explanation that clarifies why the text is ironic. For these text + explanation pairs, your task is to identify and extra background knowledge that is not present in the text itself.\n\
            This background knowledge can include common assumptions, factual knowledge and social conventions. Make sure to clearly structure the output and ensure that all knowledge is explicitly mentioned in the explanation.\n"

        # append the examples for each unique label in the prompt template
        for samplenr, sample in lable_samples.iterrows():
            example_input = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , sample['text'])\
            .replace("{PLACEHOLDER_FOR_EXPLANATION}", sample['explanation'])
            example_knowledge = ""
            if sample['commonsense'] == "[]" or sample['commonsense'] == np.nan:
                cs_annotation = "['No knowledge found']"
                cs_annotation = ast.literal_eval(cs_annotation)
            else:
                cs_annotation = ast.literal_eval(sample['commonsense'])
            # read in as list 
            # for each common sense node, add it to the 
            for cs_index, cs_entry in enumerate(cs_annotation):
                example_knowledge = f"{example_knowledge}\n{cs_index+1}. {cs_entry}"
            example_input = example_input.replace("{PLACEHOLDER_FOR_KNOWLEDGE}", example_knowledge)
            user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', example_input, re.DOTALL)
            system_text = re.search(r'\\begin\[assistant\](.*?)\\end\[assistant\]', example_input, re.DOTALL)
            if samplenr == 0:
                chat.append({"role": "user", "content": systemprompt + user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ")})
            else:
                chat.append({"role": "user", "content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ", " ")})
            chat.append({"role": "assistant", "content": system_text.group(0).replace(r"\begin[assistant]", "").replace(r"\end[assistant]", "").replace("  ", " ")},)

        # get the actual to classify text
        desired_output = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , row['text'])\
        .replace("{PLACEHOLDER_FOR_EXPLANATION}", row['labels'])

        sample_knowledge = ""
        if row['commonsense'] == "[]" or row['commonsense'] == np.nan:
            cs_annotation = "['No knowledge found']"
            cs_annotation = ast.literal_eval(cs_annotation)
        else:
            cs_annotation = ast.literal_eval(row['commonsense'])
        
        for cs_index, cs_entry in enumerate(cs_annotation):
            sample_knowledge = f"{sample_knowledge}\n{cs_index+1}. {cs_entry}"
            
        desired_output = desired_output.replace("{PLACEHOLDER_FOR_KNOWLEDGE}", sample_knowledge)

        # get text between \begin[user] and \end[user]
        user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', desired_output, re.DOTALL)
        # get text for system , this part is ignored for testing/inference only used as gold for training
        system_text = re.search(r'\\begin\[assistant\](.*?)\\end\[assistant\]', desired_output, re.DOTALL)

        chat.append({"role": "user",
                     "content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  ",

                                                                                                    " ")})
        # when training, the system needs to be provided with the gold label for SFT (SUPERVISED)
        # for testing, the system should end with the user input turn and provide the input for the assistant
        if train == True:
            chat.append({"role": "assistant",
                     "content": system_text.group(0).replace(r"\begin[assistant]", "").replace(r"\end[assistant]",
                                                                                               "").replace("  ",
                                                                                                           " ")}, )
        # this function applies the correct user start of turn, end of turn and end of sequence tokens
        # this is essential for instruction-tuned models
        desired_output = tokenizer.apply_chat_template(chat, tokenize=False)
        # print(desired_output)
        # append this complete history as desired output
        full_explanations.append(desired_output)
    #add to dataframe
    dataframe["desired_output"] = full_explanations
    return dataframe

def main():
    language = "EN"
    task = "knowledge"
    access_token = "hf_placeholder"
    huggingface_hub.login(token=access_token)
    model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
    save_as_name = model_name.split("/")[1]
    
    trained_model = f"finetuned_models/{save_as_name}_{task}_{language}"
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(trained_model, quantization_config=bnb_config, token=access_token )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    prompt = r"\begin[user]Extract the background knowledge for this text + explanation pair:\n### Text: {PLACEHOLDER_FOR_INPUTTEXT}\n### Explanation:{PLACEHOLDER_FOR_EXPLANATION}\end[user]\begin[assistant]### Extracted knowledge:{PLACEHOLDER_FOR_KNOWLEDGE}\end[assistant]"
    train_data = pd.read_csv(f'data/{language}/train.csv')

    # if the label is a string, lowercase the label
    
    train_data["labels"] = train_data["explanation"] #.apply(lambda x: x.lower() if type(x) == str else x)

    test_data = pd.read_csv(f'data/{language}/test.csv')
    test_data["labels"] = test_data["explanation"] #.apply(lambda x: x.lower() if type(x) == str else x)
    train_data = Set_Prompt_Template(dataframe=train_data, example_df=train_data, prompt=prompt, tokenizer=tokenizer, train=True)
    test_data = Set_Prompt_Template(dataframe=test_data, example_df=train_data, prompt=prompt, tokenizer=tokenizer, train=False)
    testset = datasets.Dataset.from_pandas(test_data)

    print("Input example")
    print(test_data["desired_output"].to_list()[0])
    generated_outputs = []
    pipe = transformers.pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_new_tokens=200)
    
    for out in pipe(KeyDataset(testset, "desired_output"), batch_size=1):
        print(out)
        generated_outputs.append(out)
    
    print(generated_outputs)
    test_data["generated_output"] = generated_outputs
    test_data.to_csv(f"data/generative_test_output_{save_as_name}_{task}_{language}.csv", index=False)


if __name__ == "__main__":
    main()
    print("Complete")