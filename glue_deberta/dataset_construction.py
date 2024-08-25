import numpy as np
import os
import pickle
import re
from tqdm import tqdm
tqdm.pandas()

import json
import pandas as pd

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
            last_tree_path[-len(tree_path):] = tree_path
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


def get_disco_features(text_key, span, disco_trees, feats_dict, use_paths=True, max_path_len=7):
    parsed = disco_trees[text_key]
    edu_nums_intersect = []
    for i in range(len(parsed['aligned_spans']) - 1):
        inter_len = fid_intersection_len(span, (parsed['aligned_spans'][i], parsed['aligned_spans'][i+1]))
        if inter_len > min((span[1] - span[0]) // 2, 2):
            edu_nums_intersect.append(i)
    feats = np.zeros(len(list(feats_dict.values())[0]))
    if len(edu_nums_intersect) > 0:
        if use_paths:
            # path feats (1/-1) only from the first edu
            path_feats = np.array(feats_dict[(text_key, edu_nums_intersect[0])])[1:max_path_len+1]
        for edu_num in edu_nums_intersect:
            feats += np.array(feats_dict[(text_key, edu_num)])
        feats /= len(edu_nums_intersect)
        if use_paths:
            feats[1:max_path_len+1] = path_feats
    feats = list(feats)
    return feats


def to_binary(lab_list, label2id):
    res = np.zeros(len(label2id))
    for lab in lab_list:
        if lab in label2id:
            res[label2id[lab]] = 1
    return res


def construct_semeval_dataset(part,
                              disco_depth=2,
                              use_rels=True,
                              use_nucsat=True,
                              use_paths=True,
                              max_path_len=7):
    
    with open('glue_deberta/labels_list.pkl', 'rb') as f:
        label_list = pickle.load(f)
    label2id = {l: i for i, l in enumerate(label_list)}
    
    articles_paths = sorted(os.listdir(os.path.join(data_path, f'{part}-articles-subtask-3/')))
    articles_content = {}
    for article_path in articles_paths:
        with open(os.path.join(data_path, f'{part}-articles-subtask-3/', article_path), 'r') as f:
            article_text = f.read().strip()
            articles_content[article_path] = article_text
    labels_all = pd.read_csv(os.path.join(data_path, f'{part}-labels-subtask-3.txt'), sep='\t', header=None).fillna('O')
    
    df = []
    for key in tqdm(articles_content):
        context_full = articles_content[key].split('\n')
        for _, i, labels in labels_all[labels_all[0] == int(key[7:-4])].values:
            df.append({
                'text': key,
                'sentence1': context_full[i-1],
                'label': to_binary(labels.split(','), label2id),
                'i': i,
                'span': (len('\n'.join(context_full[:i-1])), len('\n'.join(context_full[:i])))})
    df = pd.DataFrame(df)
    
    with open(f'datasets/discourse_trees/full_parsed_result_{part}_matched.pkl', 'rb') as f:   
        disco_trees = pickle.load(f)
    
    feats_dict = construct_disco_features_by_spans(disco_trees, disco_depth=disco_depth, use_rels=use_rels,
                                                   use_nucsat=use_nucsat, use_paths=use_paths, max_path_len=max_path_len)
    df['feature'] = df[['text', 'span']].progress_apply(lambda x: get_disco_features(x[0], x[1], disco_trees, feats_dict,
                                                                                    use_paths=use_paths, max_path_len=max_path_len), axis=1)
    
    num_feats = len(df['feature'].values[0])
    
    df.to_json(f'datasets/deberta_propaganda_classif/{part}_custom_feats={num_feats}_lvl{disco_depth}--use_rels={use_rels}--use_nucsat={use_nucsat}--use_paths={use_paths}_multi.json', orient='records')
    
        