 
import argparse
from trainer import *
from main import *
from utils import *
from data_loader import *

import pickle
def load_data_(args, mode):

    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
    # tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens())
    trainer.load_model()
    trainer.evaluate(mode)
    intent_list = get_intent_labels(args)

    return tokenizer, intent_list, train_dataset, dev_dataset, test_dataset
 
logger = logging.getLogger()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--mode", default='test',  type=str, help="mode is selected in {dev, test}")

    # args = parser.parse_args([e for e in " --task atis --model_type bert --model_dir atis_models/ep30  --do_eval".split() if len(e) > 0])
    args_err_analyze = parser.parse_args()
    mode = args_err_analyze.mode

    logging.basicConfig(level=logging.INFO, filename='{}/{}.err.log'.format(args_err_analyze.model_dir, mode),
                            filemode='w',)

    args_model = torch.load(f"{args_err_analyze.model_dir}/training_args.bin")
    if not hasattr(args_model, 'combine_local_context'):
        setattr(args_model, 'combine_local_context', False)
    if not hasattr(args_model, 'no_tensorboard'):
        setattr(args_model, 'no_tensorboard', True)
    args_model.model_dir = args_err_analyze.model_dir
        
    
    tokenizer, intent_list, train_dataset, dev_dataset, test_dataset = load_data_(args_model, mode)
    data_checking = test_dataset if mode == "test" else dev_dataset

    intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list, total_result = \
        pickle.load(open(f"{args_err_analyze.model_dir}/{mode}.output.pkl", 'rb'))
    total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list, 
                    classification_detail_report=True, logger=logger) 

    # Get the intent comparison result
    intent_result = (intent_preds == out_intent_label_ids)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds_list, out_slot_label_list):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result)
    false_sample_idx = [i for i, e in enumerate(sementic_acc) if not e]



    for idx in false_sample_idx:
        logger.info("=======")
        logger.info("id={}".format(idx))
        words = " ".join([e for e in tokenizer.convert_ids_to_tokens(data_checking[idx][0]) if e!= "[PAD]"][1:-1]).replace(" ##", "").split(" ")
        logger.info("sentence  = {}".format(words))
        logger.info("slot_pred = {}".format([f"{lb}[{w}]" for lb, w in zip(slot_preds_list[idx], words)]))
        logger.info("slot_gold = {}".format([f"{lb}[{w}]" for lb, w in zip(out_slot_label_list[idx], words)]))
        logger.info("intt_pred = {}".format(intent_list[intent_preds[idx]]))
        logger.info("intt_gold = {}".format(intent_list[out_intent_label_ids[idx]]))

    logger.info("=====\n final result".format(total_result))
    logger.info("{}".format(total_result))
 