from ner_deberta_multi.modeling_custom_deberta_v2 import DebertaV2BiLSTMCRFForTokenClassification
from ner_deberta_multi.custom_configuration_deberta_v2 import DebertaV2Config
from transformers import DebertaV2TokenizerFast

from accelerate import Accelerator
from datasets import load_dataset, ClassLabel
import evaluate
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from tqdm import tqdm


def get_lab_names(lab_list, label_list):
    lab_names = [label_list[i] for i in range(len(lab_list)) if lab_list[i]]
    return lab_names if len(lab_names) > 0 else ['O'] 

def get_labels(predictions, references, label_list, device):
        # Transform predictions and references tensos to numpy arrays
        if device == "cpu":
            #y_pred = predictions.detach().clone().numpy() # already numpy for CRF
            y_pred = predictions
            y_true = references.detach().clone().numpy()
        else:
            #y_pred = predictions.detach().cpu().clone().numpy() # already numpy for CRF
            y_pred = predictions
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [get_lab_names(p, label_list) if l[0] != -100 else ['O'] for (p, l) in zip(pred, gold_label)]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [get_lab_names(l, label_list) if l[0] != -100 else ['O'] for (p, l) in zip(pred, gold_label)]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels


def run_inference(model_name_or_path, dev_json, labels_list='ner_deberta/labels_list_binary.pkl', device='cuda', thresh=0.5):
    config = DebertaV2Config.from_pretrained(model_name_or_path)
    model = DebertaV2BiLSTMCRFForTokenClassification.from_pretrained(model_name_or_path, config=config).to(device).train(False)
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name_or_path, use_fast=True)
    num_labels = len(model.config.id2label)
    label_list = list(model.config.label2id.keys())
    raw_datasets = load_dataset('json', data_files={'dev': dev_json})
            
    accelerator = Accelerator()
    # Preprocessing the datasets.
    label_all_tokens = False
    
    # First we tokenize all the texts.
    padding = "max_length"
    text_column_name = 'token'   #tokens
    label_column_name = 'label'   #tokens_tags
    
    with_feats = 'feature' in raw_datasets["dev"].column_names
    feature_column_name = 'feature' if with_feats else label_column_name
    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=128,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        
        words = []
        labels = []
        features = []
        for i, (label, feature) in enumerate(zip(examples[label_column_name], examples[feature_column_name])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            feature_ids = []
            words_ids_cur = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                words_ids_cur.append(word_idx)
                if word_idx is None:
                    feature_ids.append([0 for _ in range(config.extra_feature_size)])
                    label_ids.append([-100 for _ in range(num_labels)])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    feature_ids.append(feature[word_idx])
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    feature_ids.append(feature[word_idx])
                    if label_all_tokens:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append([-100 for _ in range(num_labels)])
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            words.append(words_ids_cur)
            features.append(feature_ids)
                
        tokenized_inputs["labels"] = labels
        tokenized_inputs['words'] = words
        if with_feats:
            tokenized_inputs["extra_token_features"] = features
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["dev"].column_names,
            desc="Running tokenizer on dataset",
        )
        
        
    dev_dataset = processed_raw_datasets["dev"]
    words_ids = [el['words'] for el in dev_dataset]
    dev_dataset = dev_dataset.remove_columns('words')
    
    raw_dataset = raw_datasets["dev"]
    
    dataloader = DataLoader(dev_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=8)
    
    preds_all, refs_all = [], []
    for step, batch in enumerate(dataloader):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            outputs = model(**batch)

        predictions = (torch.sigmoid(outputs.logits) > thresh).long() # TODO check different thresholds?

        labels = batch["labels"]

        predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions_gathered = predictions_gathered[: len(eval_dataloader.dataset) - samples_seen]
                labels_gathered = labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += labels_gathered.shape[0]

        preds, refs = get_labels(predictions_gathered, labels_gathered, label_list, device)
        preds_all.extend(preds)
        refs_all.extend(refs)
        
    span_preds = []
    for j in tqdm(range(len(preds_all))):
        word2label = {}
        for i in range(len(preds_all[j])):
            if words_ids[j][i] is None or words_ids[j][i] in word2label:
                continue
            word2label[words_ids[j][i]] = preds_all[j][i]
        span_preds_cur = []
        for w in word2label:
            if word2label[w] != 0:
                span_preds_cur.append((raw_dataset[j]['sent'][w], raw_dataset[j]['span'][w], word2label[w]))
        span_preds.append(span_preds_cur)
        
    pred_df = pd.DataFrame(sum(span_preds, []), columns=['article', 'span', 'pred'])
    return pred_df
