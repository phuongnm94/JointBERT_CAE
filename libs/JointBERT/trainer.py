import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import pickle

from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_by2task_labels, get_slot_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        
        self.writer = SummaryWriter(log_dir=self.args.model_dir+"/tensorboard/")
        
        self.intent_label_lst = get_intent_labels(args)
        org_slot_labels = get_slot_labels(args)
        if "slotby2task" in self.args.task:
            self.slot_label_lst, self.slot_type_label_lst, self.org_slot_labels = get_slot_by2task_labels(args)
        else:
            self.slot_label_lst, self.slot_type_label_lst, self.org_slot_labels = org_slot_labels, [], org_slot_labels

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst,
                                                      slot_type_label_list=self.slot_type_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_type_labels_ids': batch[5] if len(batch) > 5 else None,
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev", global_step)
                        self.evaluate("test", global_step)

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        
        return global_step, tr_loss / global_step

    @staticmethod    
    def recover_slot_labelby2task(slot_labels, slot_type_labels, slot_label_map, slot_type_label_map, slot_gold_labels):
        start_object_id = [k for k, v in slot_label_map.items() if v == 'B-object'][0]
        in_object_id = [k for k, v in slot_label_map.items() if v == 'I-object'][0]
        out_object_id = [k for k, v in slot_label_map.items() if v == 'O'][0]
        pad_id = [k for k, v in slot_label_map.items() if v == 'PAD'][0]
        out_object_type_id = [k for k, v in slot_type_label_map.items() if v == 'O'][0]

        for i in range(slot_labels.shape[0]):
            for j in range(slot_labels.shape[1]):
                if slot_labels[i, j] == start_object_id:
                    for k in range(j+1, slot_labels.shape[1]):
                        if slot_gold_labels[i, k] == pad_id: # skeep label of subwords
                            continue
                        if slot_labels[i, k] == in_object_id:
                            slot_type_labels[i, k] = slot_type_labels[i, j] # slot_type_labels[i, j] => cur_obj_id
                        else:
                            break
                elif slot_labels[i, j] == out_object_id:
                    slot_type_labels[i, j] = out_object_type_id

        return slot_type_labels

    def evaluate(self, mode, global_step=None):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None
        out_slot_type_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_type_labels_ids': batch[5] if len(batch) > 5 else None,
                          'slot_labels_ids': batch[4]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                if "slotby2task" in self.args.task: 
                    tmp_eval_loss, (intent_logits, slot_logits, slot_type_logits) = outputs[:2]
                else:
                    tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
                    slot_type_logits = None

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if out_intent_label_ids is None:
                if '_ner_' not in self.args.model_type: # skip intent prediction in the case using NER model 
                    intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                if '_ner_' not in self.args.model_type:  # skip intent prediction in the case using NER model 
                    intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    if not "slotby2task" in self.args.task:
                        # decode() in `torchcrf` returns list with best index directly
                        slot_preds = np.array(self.model.crf.decode(slot_logits))
                    else:
                        start_obj_id = self.slot_label_lst.index('B-object')
                        tmp_slot_preds = torch.argmax(slot_logits, dim=2)
                        mask_object_pred = tmp_slot_preds == start_obj_id
                        mask_object_pred[:, 0] = False #warranty that first token alway not object entity

                        slot_type_new_logits, slot_type_new_mask, _ = self.model.reconstruct_entity_logits(
                            slot_type_logits, mask_object_pred, slot_type_labels_ids=None)

                        # combine sample in batch 
                        slot_preds = slot_logits.detach().cpu().numpy()
                        slot_type_preds = self.model.crf.decode(slot_type_new_logits, mask=slot_type_new_mask)
                        slot_type_preds = self.model.place_entity_by_mask(slot_type_preds, mask_object_pred,
                                                                          padding_value=self.args.ignore_index).detach().cpu().numpy()                
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                    slot_type_preds = slot_type_logits.detach().cpu().numpy() if "slotby2task" in self.args.task else None

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
                out_slot_type_labels_ids = inputs["slot_type_labels_ids"].detach().cpu().numpy() if "slotby2task" in self.args.task else None
            else:
                if self.args.use_crf:
                    if not "slotby2task" in self.args.task:
                        slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                    else:
                        start_obj_id = self.slot_label_lst.index('B-object')
                        tmp_slot_preds = torch.argmax(slot_logits, dim=2)
                        mask_object_pred = tmp_slot_preds == start_obj_id
                        mask_object_pred[:, 0] = False #warranty that first token alway not object entity

                        slot_type_new_logits, slot_type_new_mask, _ = self.model.reconstruct_entity_logits(
                            slot_type_logits, mask_object_pred, slot_type_labels_ids=None)
                        slot_type_preds_new = self.model.crf.decode(slot_type_new_logits, mask=slot_type_new_mask)
                        slot_type_preds_new = self.model.place_entity_by_mask(slot_type_preds_new, mask_object_pred,
                                                                              padding_value=self.args.ignore_index).detach().cpu().numpy()         

                        # combine sample in batch 
                        slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                        slot_type_preds = np.append(slot_type_preds, slot_type_preds_new, axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                    slot_type_preds = np.append(slot_type_preds, slot_type_logits.detach().cpu().numpy(), axis=0) if "slotby2task" in self.args.task else None

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)
                out_slot_type_labels_ids = np.append(out_slot_type_labels_ids, inputs["slot_type_labels_ids"].detach().cpu().numpy(), axis=0) if "slotby2task" in self.args.task else None

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        if '_ner_' not in self.args.model_type: 
            intent_preds = np.argmax(intent_preds, axis=1)
        else: 
            # skip intent prediction in the case using NER model 
            intent_preds = out_intent_label_ids

        # Slot result
        if "slotby2task" not in self.args.task:
            if not self.args.use_crf:
                slot_preds = np.argmax(slot_preds, axis=2)
            slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
            out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
            slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

            for i in range(out_slot_labels_ids.shape[0]):
                for j in range(out_slot_labels_ids.shape[1]):
                    if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                        out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                        slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
        else:
            if not self.args.use_crf:
                slot_preds = np.argmax(slot_preds, axis=2)
                slot_type_preds = np.argmax(slot_type_preds, axis=2)
            else:
                # using crf on classify entity type
                slot_preds = np.argmax(slot_preds, axis=2)

            slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
            slot_type_label_map = {i: label for i, label in enumerate(self.slot_type_label_lst)}
            out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
            slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
            
            out_slot_type_labels_ids = self.recover_slot_labelby2task(out_slot_labels_ids, out_slot_type_labels_ids, slot_label_map, slot_type_label_map, out_slot_labels_ids)
            slot_type_preds = self.recover_slot_labelby2task(slot_preds, slot_type_preds, slot_label_map, slot_type_label_map, out_slot_labels_ids)
            for i in range(out_slot_labels_ids.shape[0]):
                for j in range(out_slot_labels_ids.shape[1]):
                    if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                        out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]].replace("object", slot_type_label_map[out_slot_type_labels_ids[i][j]]))
                        cur_slot_type_pred = slot_type_preds[i][j]
                        if cur_slot_type_pred >= len(slot_type_label_map):
                            cur_slot_type_pred = 0
                        slot_preds_list[i].append(slot_label_map[slot_preds[i][j]].replace("object", slot_type_label_map[cur_slot_type_pred]))

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)
        
        # only dump pickle result when not in training process
        if  global_step is None or global_step==0:
            pickle.dump((intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list, total_result), 
                    open('{}/{}.output.pkl'.format(self.args.model_dir, mode), 'wb'))

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        
        if not self.args.no_tensorboard and global_step is not None:
            for key in sorted(results.keys()):
                self.writer.add_scalar(f"{mode}/{key}", results[key], global_step)

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        # try:
        self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                        args=self.args,
                                                        intent_label_lst=self.intent_label_lst,
                                                        slot_label_lst=self.slot_label_lst,
                                                        slot_type_label_list=self.slot_type_label_lst)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")
        # except:
        #     raise Exception("Some model files might be missing...")
