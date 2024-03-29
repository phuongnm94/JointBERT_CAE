import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from transformers import BertConfig, DistilBertConfig, AlbertConfig, RobertaConfig, AutoTokenizer
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import JointBERT, JointDistilBERT, JointAlbert, JointBERTSlotby2task, JointRoBERTa, JointRoBERTaSlotby2task, BERTner, BERTnerSlotby2task

MODEL_CLASSES = {
    'phobert': (RobertaConfig, JointRoBERTa, AutoTokenizer),
    'phobert-seqlabelby2task': (RobertaConfig, JointRoBERTaSlotby2task, AutoTokenizer),
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'bertLcased_ner_': (BertConfig, BERTner, BertTokenizer),
    'bertLcased-seqlabelby2task_ner_': (BertConfig, BERTnerSlotby2task, BertTokenizer),
    'bertcased_ner_': (BertConfig, BERTner, BertTokenizer),
    'bertcased-seqlabelby2task_ner_': (BertConfig, BERTnerSlotby2task, BertTokenizer),
    'bertcased': (BertConfig, JointBERT, BertTokenizer),
    'bertcased-seqlabelby2task': (BertConfig, JointBERTSlotby2task, BertTokenizer),
    'bertseqlabelby2task': (BertConfig, JointBERTSlotby2task, BertTokenizer),
    'bertseqlabelby2task': (BertConfig, JointBERTSlotby2task, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'phobert': 'vinai/phobert-base',
    'phobert-seqlabelby2task': 'vinai/phobert-base',
    'bertLcased_ner_': 'bert-large-cased',
    'bertLcased-seqlabelby2task_ner_': 'bert-large-cased',
    'bertcased_ner_': 'bert-base-cased',
    'bertcased-seqlabelby2task_ner_': 'bert-base-cased',
    'bertcased': 'bert-base-cased',
    'bertcased-seqlabelby2task': 'bert-base-cased',
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'bertseqlabelby2task': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

def get_slot_by2task_labels(args):
    org_slot_labels = get_slot_labels(args)
    default_lb =  [e for e in org_slot_labels if not (e.startswith("B-") or e.startswith("I-"))]
    slot_label_lst = default_lb + ["B-object", "I-object"]
    real_type = [] 
    for ii_check in [e for e in org_slot_labels if e.startswith("B-") or e.startswith("I-")]:
        if ii_check[2:] not in real_type:
            real_type.append(ii_check[2:])
    slot_type_label_lst = default_lb + real_type
    return slot_label_lst, slot_type_label_lst, org_slot_labels


def load_tokenizer(args):
    if "phobert" in args.model_type:
        return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path, use_fast=False)
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels, classification_detail_report=False, logger=None):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    if classification_detail_report:
        logger.info(classification_report(slot_labels, slot_preds, digits=4))
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)
    
    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }
