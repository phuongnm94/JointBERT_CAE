 
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

def load_data(mode):

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    # args = parser.parse_args([e for e in " --task atis --model_type bert --model_dir atis_models/ep30  --do_eval".split() if len(e) > 0])
    args = parser.parse_args([e for e in " --task atis_slotby2task --model_type bertseqlabelby2task --model_dir atis_models/ep30-dev-slot_by2task  --do_eval".split() if len(e) > 0])

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    return load_data_(args, mode)

logger = logging.getLogger(__name__)


if __name__=="__main__":
    mode = "dev"
    tokenizer, intent_list, train_dataset, dev_dataset, test_dataset = load_data(mode)
    data_checking = test_dataset if mode == "test" else dev_dataset

    intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list, total_result = \
        pickle.load(open('atis_models/ep30-dev-slot_by2task/{}.output.pkl'.format(mode), 'rb'))
    total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list) 

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

    logging.basicConfig(level=logging.INFO, filename='atis_models/ep30-dev-slot_by2task/{}.err.log'.format(mode),
                            filemode='w',)

    for idx in false_sample_idx:
        logger.info("=======")
        logger.info("id={}".format(idx))
        logger.info("sentence  = {}".format([e for e in tokenizer.convert_ids_to_tokens(data_checking[idx][0]) if e!= "[PAD]"]))
        logger.info("slot_pred = {}".format(slot_preds_list[idx]))
        logger.info("slot_gold = {}".format(out_slot_label_list[idx]))
        logger.info("intt_pred = {}".format(intent_list[intent_preds[idx]]))
        logger.info("intt_gold = {}".format(intent_list[out_intent_label_ids[idx]]))

    logger.info("=====\n final result".format(total_result))
    logger.info("{}".format(total_result))
 