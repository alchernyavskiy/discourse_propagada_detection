from modeling_custom_deberta_v2 import DebertaV2ForSequenceClassification
from custom_configuration_deberta_v2 import DebertaV2Config
from transformers import DebertaV2Tokenizer

from accelerate import Accelerator
from datasets import load_dataset, ClassLabel
import evaluate
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm


def run_inference(model_name_or_path, dev_json, device='cuda'):
    config = DebertaV2Config.from_pretrained(model_name_or_path)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_name_or_path, config=config).to(device).train(False)
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name_or_path, use_fast=True)
    
    raw_datasets = load_dataset('json', data_files={'dev': dev_json})
            
    accelerator = Accelerator()
    padding = "max_length"
    sentence1_key, sentence2_key = "sentence1", None#"sentence2"
    max_length = 256

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
        
        if "feature" in examples:
            result["extra_features"] = examples["feature"]
        
        return result

    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["dev"].column_names,
            desc="Running tokenizer on dataset",
        )
          
    dev_dataset = processed_raw_datasets["dev"]
    dataloader = DataLoader(dev_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=8)
    
    preds = []
    model.eval();
    for step, batch in tqdm(enumerate(dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            outputs = model(**batch)
        #predictions = outputs.logits.argmax(dim=-1)
        predictions = (torch.sigmoid(outputs.logits) > 0.5).long()
        preds.extend([[model.config.id2label[i] for i in range(len(p)) if p[i] == 1] for p in predictions])
        
    return preds
