import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert  import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import BridgeIntentEntities, IntentClassifier, NgramMLP, SlotClassifier, NgramLSTM


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst, **kwargs):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)
        if args.combine_local_context:
            self.local_context = NgramMLP(4, config.hidden_size)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        if self.args.combine_local_context:
            tmp = sequence_output.clone()[:, 1:, :]
            sequence_output = sequence_output.clone()
            sequence_output[:, 1:, :] =  self.local_context(tmp)
            slot_logits = self.slot_classifier(sequence_output)
        else:
            slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

class JointBERTSlotby2task(JointBERT):
    def __init__(self, config, args, intent_label_lst, slot_label_lst, slot_type_label_list):
        super(JointBERTSlotby2task, self).__init__(config, args, intent_label_lst, slot_label_lst) 
        self.num_slot_type_labels = len(slot_type_label_list)
        self.slot_type_classifier = SlotClassifier(config.hidden_size, self.num_slot_type_labels, args.dropout_rate)
        if args.combine_local_context:
            self.local_context = NgramLSTM(4, config.hidden_size, args.dropout_rate)
        
        # override crf layer 
        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_type_labels, batch_first=True)
    
    @staticmethod
    def reconstruct_entity_logits(slot_type_logits, entity_masked, slot_type_labels_ids=None):
        count_entity = torch.sum(entity_masked, dim=1)
        slot_type_new_logits = torch.zeros_like(slot_type_logits)
        slot_type_new_mask= torch.zeros_like(entity_masked).fill_(False)
        slot_type_new_labels_ids = torch.zeros_like(slot_type_labels_ids) if slot_type_labels_ids is not None else None

        for i_sample in range(slot_type_logits.shape[0]):
            found_entity_count = 0
            for i_w in range(slot_type_logits.shape[1]):
                if entity_masked[i_sample][i_w] or i_w == 0:
                    slot_type_new_logits[i_sample, found_entity_count] = slot_type_logits[i_sample, i_w]
                    slot_type_new_mask[i_sample, found_entity_count] = True
                    if slot_type_new_labels_ids is not None:
                        slot_type_new_labels_ids[i_sample, found_entity_count] = slot_type_labels_ids[i_sample, i_w]
                    found_entity_count += 1
            assert found_entity_count == count_entity[i_sample] + 1
        return slot_type_new_logits, slot_type_new_mask, slot_type_new_labels_ids

    @staticmethod
    def place_entity_by_mask(slot_type_preds, masked, padding_value=0):
        new_slot_type_preds = torch.IntTensor(masked.shape).fill_(padding_value).to(masked.device)
        for i_sample in range(masked.shape[0]):
            i_entity = 1                            # skip first entity
            for i_w in range(1, masked.shape[1]):   # skip first token CLS
                if masked[i_sample, i_w]:
                    new_slot_type_preds[i_sample, i_w] = slot_type_preds[i_sample][i_entity]
                    i_entity += 1
            assert i_entity == len(slot_type_preds[i_sample])
        return new_slot_type_preds

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, slot_type_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        if self.args.combine_local_context:
            tmp = sequence_output.clone()[:, 1:, :]
            sequence_output = sequence_output.clone()
            sequence_output[:, 1:, :] = 0.5*(tmp + self.local_context(tmp)) 
            slot_type_logits = self.slot_type_classifier(sequence_output)
        else:
            slot_type_logits = self.slot_type_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax + slot type softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                # loss for type of Anonymous entities
                entity_masked = slot_type_labels_ids.ne(0)

                slot_type_new_logits, slot_type_new_mask, slot_type_new_labels_ids = self.reconstruct_entity_logits(
                    slot_type_logits, entity_masked, slot_type_labels_ids)

                slot_loss = self.crf(slot_type_new_logits, slot_type_new_labels_ids, mask=slot_type_new_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood

                # loss for Anonymous entity detection 
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss] 
                    active_labels = slot_labels_ids.view(-1)[active_loss] 
                    slot_loss += slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss += slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1

                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_slot_type_logits = slot_type_logits.view(-1, self.num_slot_type_labels)[active_loss]

                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    active_slot_type_labels = slot_type_labels_ids.view(-1)[active_loss]

                    slot_loss = slot_loss_fct(active_logits, active_labels) + slot_loss_fct(active_slot_type_logits, active_slot_type_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1)) \
                        + slot_loss_fct(slot_type_logits.view(-1, self.num_slot_labels), slot_type_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss


        outputs = ((intent_logits, slot_logits, slot_type_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
