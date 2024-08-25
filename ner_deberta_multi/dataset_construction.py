import numpy as np
import os
import pickle
import re
from tqdm import tqdm
tqdm.pandas()

import json
import pandas as pd
import spacy
from spacy.training import offsets_to_biluo_tags

import portion as P
from functools import reduce
from transformers import DebertaV2TokenizerFast

data_path = 'SemEval-2023-task3-corpus-main/data/en/'

labels_list = ['Appeal_to_Authority',
 'Appeal_to_Fear-Prejudice',
 'Appeal_to_Hypocrisy',
 'Appeal_to_Popularity',
 'Causal_Oversimplification',
 'Conversation_Killer',
 'Doubt',
 'Exaggeration-Minimisation',
 'False_Dilemma-No_Choice',
 'Flag_Waving',
 'Guilt_by_Association',
 'Loaded_Language',
 'Name_Calling-Labeling',
 'Obfuscation-Vagueness-Confusion',
 'Red_Herring',
 'Repetition',
 'Slogans',
 'Straw_Man',
 'Whataboutism']


disco_labels = ['Attribution',
 'Background',
 'Cause',
 'Comparison',
 'Condition',
 'Contrast',
 'Elaboration',
 'Enablement',
 'Evaluation',
 'Explanation',
 'Joint',
 'Manner-Means',
 'Same-Unit',
 'Summary',
 'Temporal',
 'Textual-Organization',
 'Topic-Change',
 'Topic-Comment',
 'span']
disco2int = {l: i for i,l in enumerate(disco_labels)}


nlp = spacy.load("en_core_web_sm")


def join_to_binary(annot, label):
    res = []
    if len(annot) > 0:
        portion_ints = [P.open(el[1], el[2]) for el in annot]
        union = reduce(lambda x, y: x | y, portion_ints)
        for s in P.to_data(union):
            res.append([label, s[1], s[2]])
    return res


def bioul_to_raw(lab):
    return 0 if lab in ['-', 'O'] else 1


def get_info_path(dt, leaf_num):
    perlevel_info = {}
    count_by_levels = {}
    
    for row in dt.split('\n'):
        level = (len(row) - len(row.lstrip())) // 2
        if 'rel2par' in row:
            if level in count_by_levels:
                count_by_levels[level] += 1
            else:
                count_by_levels[level] = 1
            perlevel_info[level] = [row.strip().split()[1], re.findall(r'rel2par ([^\)]*)\)', row)[0]]
        leaf_num_cur = re.findall(r'\(leaf (\d+)\)', row)
        if len(leaf_num_cur) > 0:
            leaf_num_cur = int(leaf_num_cur[0])
            if leaf_num_cur == leaf_num:
                path = [count_by_levels[l] % 2 * (-2) + 1 for l in range(1, level+1)]
                return ({l: v for (l,v) in perlevel_info.items() if l <= level}, path)
    
    
def construct_disco_features_by_spans(disco_trees, disco_depth=3, use_rels=True, use_nucsat=True,
                                      use_paths=False, max_path_len=7):
    feats_result = {}
    for text_key in tqdm(disco_trees):
        for i in range(len(disco_trees[text_key]['aligned_spans']) - 1):
            tree_info, tree_path = get_info_path(disco_trees[text_key]['dt'], i+1) # edu num starts from 1 in trees
            last_disco = list(tree_info.values())[-disco_depth:]
            last_tree_path = [0 for _ in range(max_path_len)]
            tree_path = tree_path[:max_path_len]
            last_tree_path[-len(tree_path):] = tree_path  # TODO: or max_path length + pad?
            last_tree_path = [i+1] + last_tree_path

            res_feats = [] if not use_paths else last_tree_path
            for _ in range(disco_depth - len(last_disco)):
                if use_rels:
                    if use_nucsat:
                        res_feats.extend([0 for _ in range(1+len(disco_labels))])
                    else:
                        res_feats.extend([0 for _ in range(len(disco_labels))])
                else:
                    if use_nucsat:
                        res_feats.extend([0])
                    else:
                        res_feats.extend([])
                        
            for el in last_disco:
                if use_rels:
                    if use_nucsat:
                        cur_feats = [0 for _ in range(1+len(disco_labels))]
                        cur_feats[0] = int(el[0] == 'Nucleus')
                        cur_feats[1 + disco2int[el[1]]] = 1
                    else:
                        cur_feats = [0 for _ in range(len(disco_labels))]
                        cur_feats[disco2int[el[1]]] = 1
                else:
                    cur_feats = [int(el[0] == 'Nucleus')] if use_nucsat else []
                res_feats.extend(cur_feats)
            feats_result[(text_key, i)] = res_feats
    return feats_result


def fid_intersection_len(a, b):
    (a_s, a_e), (b_s, b_e) = a, b
    if b_s > a_e or a_s > b_e:
        return 0
    return min(a_e, b_e) - max(a_s, b_s)


def check_is_edu_start_end(sp, span_nums, spans_list):
    edu_start, edu_end = spans_list[span_nums[0]], spans_list[span_nums[-1]+1]
    is_start = abs(sp[0] - edu_start) < 2
    is_end = abs(sp[1] - edu_end) < 2
    return is_start, is_end


def get_disco_features(text_key, span, disco_trees, feats_dict, use_start_end=True):
    parsed = disco_trees[text_key]
    edu_nums_intersect = []
    for i in range(len(parsed['aligned_spans']) - 1):  # IS BY WORDS!!!
        inter_len = fid_intersection_len(span, (parsed['aligned_spans'][i], parsed['aligned_spans'][i+1]))
        if inter_len > min((span[1] - span[0]) // 2, 2):
            edu_nums_intersect.append(i)
    feats = np.zeros(len(list(feats_dict.values())[0]))
    if len(edu_nums_intersect) == 0:
        edu_nums_intersect.append(len(parsed['aligned_spans']) - 2)
    for edu_num in edu_nums_intersect:
        feats += np.array(feats_dict[(text_key, edu_num)])
    feats /= len(edu_nums_intersect)
    feats = list(feats)
    if not use_start_end:
        return feats
    is_edu_start, is_edu_end = check_is_edu_start_end(span,
                                                      [edu_nums_intersect[0], edu_nums_intersect[-1]],
                                                      parsed['aligned_spans'])
    res_feats = [int(is_edu_start), int(is_edu_end)] + feats
    return res_feats


def preproc_ner_json(dataset, max_len=256):
    model_name_or_path = "microsoft/deberta-v3-base"   # CHANGE FOR THE SPECIFIC MODEL!
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name_or_path, use_fast=True)
    max_len -= tokenizer.num_special_tokens_to_add()
    
    subword_len_counter = 0
    cur_sent_num = -1
    
    result = []
    cur_res = []
    for line in tqdm(dataset[['Text', 'Token', 'Label', 'Span', 'Disco']].values):
        sent_num = line[0]
        current_subwords_len = len(tokenizer.tokenize(line[1]))

        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue

        if subword_len_counter + current_subwords_len > max_len or sent_num != cur_sent_num:
            if len(cur_res) > 0:
                cur_res_json = {
                    'token': [el[0] for el in cur_res],
                    'label': [el[1] for el in cur_res],
                    'span': [el[2] for el in cur_res],
                    'sent': [el[3] for el in cur_res],
                    'feature': [el[-1] for el in cur_res]
                }
                result.append(cur_res_json)
            subword_len_counter = current_subwords_len
            cur_sent_num = sent_num
            cur_res = [(line[1], line[2], line[3], line[0], line[4])]
            continue

        subword_len_counter += current_subwords_len
        cur_res.append((line[1], line[2], line[3], line[0], line[4]))
    
    if len(cur_res) > 0:
        cur_res_json = {
            'token': [el[0] for el in cur_res],
            'label': [el[1] for el in cur_res],
            'span': [el[2] for el in cur_res],
            'sent': [el[3] for el in cur_res],
            'feature': [el[-1] for el in cur_res]
        }
        result.append(cur_res_json)
    
    return result


def construct_semeval_dataset(part,
                              disco_depth=2,
                              use_rels=True,
                              use_nucsat=True,
                              use_start_end=True,
                              use_paths=True):
    articles_paths = sorted(os.listdir(os.path.join(data_path, f'{part}-articles-subtask-3/')))
    labels_dict = {}
    articles_content = {}

    for article_path in articles_paths:

        with open(os.path.join(data_path, f'{part}-articles-subtask-3/', article_path), 'r') as f:
            article_text = f.read().strip()
            articles_content[article_path] = article_text

        with open(os.path.join(data_path, f'{part}-labels-subtask-3-spans/',
                               article_path.replace('.txt', '-labels-subtask-3.txt')), 'r') as f:
            labels_text = f.read().strip().split('\n')

            labels_dict[article_path] = []
            for row in labels_text:
                row = row.split('\t')
                if len(row) > 1:
                    labels_dict[article_path].append([row[1], int(row[2]), int(row[3])])
                    
    df = []
    for key in tqdm(labels_dict):
        text, annot = articles_content[key], labels_dict[key]
        doc=nlp(text)
        xx = ([token.text for token in doc])
        spans = ([(token.idx, token.idx + len(token.text)) for token in doc])
        tags = []
        for label in labels_list:
            cur_labels = join_to_binary([lab for lab in annot if lab[0] == label], label)
            tags.append([bioul_to_raw(el) for el in
                         offsets_to_biluo_tags(doc, [[a[1], a[2], a[0]] for a in cur_labels])])
        df.append({'Text': key,'Token': xx, 'Span': spans, 'Label': np.array(tags).T})
    df = pd.DataFrame(df)
    
    df_full = df.explode(['Token', 'Span', 'Label'])
    df_full['Label'] = df_full['Label'].apply(list)
    
    with open(f'datasets/discourse_trees/full_parsed_result_{part}_matched.pkl', 'rb') as f:   
        disco_trees = pickle.load(f)

    feats_dict = construct_disco_features_by_spans(disco_trees, disco_depth=disco_depth, use_rels=use_rels,
                                                   use_nucsat=use_nucsat, use_paths=use_paths)
    df_full['Disco'] = df_full[['Text', 'Span']].progress_apply(lambda x:
                                            get_disco_features(x[0], x[1], disco_trees, feats_dict, use_start_end=use_start_end), axis=1)
    data_train = preproc_ner_json(df_full)
    for el in data_train:
        el['label'] = [[int(l) for l in lst] for lst in el['label']]
    
    num_feats = len(data_train[0]['feature'][0])
    
    with open(f'datasets/deberta_propaganda_full/{part}_binary_custom_feats={num_feats}_lvl{disco_depth}--use_rels={use_rels}--use_nucsat={use_nucsat}--use_paths={use_paths}--use_start_end={use_start_end}_multi.json', 'w') as outfile:
        json.dump(data_train, outfile)
        