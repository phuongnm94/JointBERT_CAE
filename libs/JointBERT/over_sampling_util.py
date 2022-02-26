    
import argparse
import enum
import io
import logging
from math import floor
import os
import random
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pickle

logger = logging.getLogger()

def draw_dist(df, path_hist='entities_hist.svg', col='support'):
    # plt.figure(figsize=(5,3))
    plt.xticks(rotation=45,   ha="right")

    sns.barplot(x="entity", y=col, data=df, color='blue')
    plt.savefig(path_hist.replace(".svg", f"_{col}.svg"),)

def random_idx_by_entity(data_lb, selected_entity):
    idx_shuff = np.random.permutation(len(data_lb))
    for idx_sample in idx_shuff:
        tk_labels = data_lb[idx_sample]
        for tk_label in tk_labels:
            if tk_label.startswith("B-") and tk_label[2:] in selected_entity:
                return idx_sample

def stats_freq_entity(label_tags):
    rp  = classification_report(label_tags, label_tags)
    rp = [l.strip() for l in rp.split("\n")[:-3]]
    rp[0] = "entity,"+rp[0]
    rp = rp[0:1] + rp[2:]
    rp = "\n".join(rp)

    rp = re.sub(r' {1,}', ',', rp)
    # print(rp)
    in_memory_file = io.StringIO(rp.strip())

    df = pd.read_csv(filepath_or_buffer=in_memory_file, sep=',', index_col='entity')
    return df
def check_arr_eq(ar1, ar2):
    for i, e in enumerate(ar1):
        if e != ar2[i]:
            return False
    return True
def update_over_sample_support(df_cur, df_append):
    for entity in df_append.index:
        appended_support = df_append.at[entity, 'support']
        df_cur.at[entity, 'over_sample_support'] = df_cur.at[entity, 'over_sample_support'] + appended_support
    return df_cur

def generate_new_data(old_data_path, new_data_path, over_sample_idxs):
    with open(f"{old_data_path}", 'rt') as f:
        data = [l.strip() for l in f.readlines()]
        new_data = data + [data[idx] for idx in over_sample_idxs]
    with open(new_data_path, 'wt') as f:
        f.write("\n".join(new_data))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--threshold", default=0.2,  type=float, help="over sample threshold, ratio different between min class with max class")

    # args = parser.parse_args([e for e in " --task atis --model_type bert --model_dir atis_models/ep30  --do_eval".split() if len(e) > 0])
    opts = parser.parse_args()
    data_dir = opts.data_dir

    with open(f"{data_dir}/train_org/seq.out", 'rt') as f:
        label_tags = [l.strip().split(" ") for l in f.readlines()]

    df = stats_freq_entity(label_tags)
    # df = df.set_index('entity')

    # total_entities = sum(df['support'])
    # df['percent_support'] = df['support'] / total_entities * 100
    df['over_sample_support'] = df['support']
    df = df.sort_values('entity')
    print(df) 

    # filter entity
    max_entity_count = max(df['support'])
    threshold = floor(max_entity_count*opts.threshold)
    selected_entity = df[df['support'] < threshold].index

    over_sample_idxs = []
    while True:
        selected_entity = df[df['over_sample_support'] < threshold].index
        if len(selected_entity) == 0:
            break

        idx_over_sample_selected = random_idx_by_entity(label_tags, selected_entity.values)
        over_sample_idxs.append(idx_over_sample_selected)
        over_sample_lb_data = [label_tags[idx_over_sample_selected]]
        stat_appended_df = stats_freq_entity(over_sample_lb_data)
        old_val = df['over_sample_support'].values
        update_over_sample_support(df, stat_appended_df)
        new_val =  df['over_sample_support'].values
        if not check_arr_eq(new_val, old_val):
            print(stat_appended_df)
            print(df)
        if len(over_sample_idxs) % 1000 == 0:
            print(len(over_sample_idxs), len(selected_entity))

    try:
        os.makedirs(f"{data_dir}/train_over_sample")
    except:
        pass
    print(df)
    df = df.reset_index()
    draw_dist(df, path_hist=f'{opts.data_dir}/train_over_sample/entities_hist.svg')
    draw_dist(df, path_hist=f'{opts.data_dir}/train_over_sample/entities_hist.svg', col='over_sample_support')

    json.dump([int(e) for e in over_sample_idxs], open(f'{opts.data_dir}/train_over_sample/idx_appended.json', 'wt', encoding='utf8'))
    # over_sample_idxs = json.load(open(f'{opts.data_dir}/train_over_sample/idx_appended.json', encoding='utf8'))

    generate_new_data(f"{data_dir}/train_org/seq.in", f"{data_dir}/train_over_sample/seq.in", over_sample_idxs)
    generate_new_data(f"{data_dir}/train_org/seq.out", f"{data_dir}/train_over_sample/seq.out", over_sample_idxs)
    generate_new_data(f"{data_dir}/train_org/label", f"{data_dir}/train_over_sample/label", over_sample_idxs)
