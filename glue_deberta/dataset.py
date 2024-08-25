import numpy as np
import os
import pickle
import re
from tqdm import tqdm
import json
import pandas as pd
import spacy

from nltk.tokenize.punkt import PunktSentenceTokenizer

import portion as P
from functools import reduce

tqdm.pandas()

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


def get_context(full_text, span, sents_around=0):
    sents_spans = list(PunktSentenceTokenizer().span_tokenize(full_text))
    sents_in = []
    for i, (start, end) in enumerate(sents_spans):
        if (span[0] >= start and span[0] < end) or \
                (span[1] >= start and span[1] < end):
            sents_in.append(i)

    if len(sents_in) == 0:
        return ""
    
    grounding_start = sents_spans[max(min(sents_in) - sents_around, 0)][0]
    grounding_end = sents_spans[min(max(sents_in) + sents_around, len(sents_spans) - 1)][1]
    return full_text[grounding_start:grounding_end]


def get_info_path(dt, leaf_num):
    perlevel_info = {}
    for row in dt.split('\n'):
        level = (len(row) - len(row.lstrip())) // 2
        if 'rel2par' in row:
            perlevel_info[level] = [row.strip().split()[1], re.findall(r'rel2par ([^\)]*)\)', row)[0]]
        leaf_num_cur = re.findall(r'\(leaf (\d+)\)', row)
        if len(leaf_num_cur) > 0:
            leaf_num_cur = int(leaf_num_cur[0])
            if leaf_num_cur == leaf_num:
                return perlevel_info
   

def construct_disco_features_by_spans(disco_trees, disco_depth=3, use_rels=True, use_nucsat=True):
    feats_result = {}
    for text_key in tqdm(disco_trees):
        for i in range(len(disco_trees[text_key]['aligned_spans']) - 1):
            tree_path = get_info_path(disco_trees[text_key]['dt'], i+1) # edu num starts from 1 in trees
            last_disco = list(tree_path.values())[-disco_depth:]

            res_feats = []
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
                    cur_feats = [int(el[0] == 'Nucleus')]
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


def construct_dataset(labels_dict, articles_content, part='dev', disco_depth=3, use_rels=True, use_nucsat=True, use_start_end=True):
    df = []
    for key in tqdm(labels_dict):
        context_full = articles_content[key]
        for cl, sp_start, sp_end in labels_dict[key]:
            sents_around = 0 if len(context_full[sp_start:sp_end].split()) < 7 else 1
            df.append({
                'text': key,
                'sentence1': context_full[sp_start:sp_end],
                'sentence2': get_context(context_full, (sp_start, sp_end), sents_around=sents_around),
                'label': cl,
                'span': (sp_start, sp_end)})
    df = pd.DataFrame(df)
    
    with open(f'datasets/discourse_trees/full_parsed_result_{part}_matched.pkl', 'rb') as f:   
        disco_trees = pickle.load(f)

    feats_dict = construct_disco_features_by_spans(disco_trees, disco_depth=disco_depth, use_rels=use_rels, use_nucsat=use_nucsat)
    
    df['feature'] = df[['text', 'span']].progress_apply(lambda x:
                                            get_disco_features(x[0], x[1], disco_trees, feats_dict, use_start_end=use_start_end), axis=1)
    return df
