import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert  import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import BridgeIntentEntities, IntentClassifier, NgramMLP, SlotClassifier, NgramLSTM


class BERTner(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst, **kwargs):
        super(BERTner, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        # if args.
        self.bert = BertModel(config=config)  # Load pretrained bert

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

        if self.args.combine_local_context:
            tmp = sequence_output.clone()[:, 1:, :]
            sequence_output = sequence_output.clone()
            sequence_output[:, 1:, :] =  self.local_context(tmp)
            slot_logits = self.slot_classifier(sequence_output)
        else:
            slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0

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

        outputs = ((pooled_output, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

class BERTnerSlotby2task(BERTner):
    def __init__(self, config, args, intent_label_lst, slot_label_lst, slot_type_label_list):
        super(BERTnerSlotby2task, self).__init__(config, args, intent_label_lst, slot_label_lst) 
        self.num_slot_type_labels = len(slot_type_label_list)
        self.slot_type_classifier = SlotClassifier(config.hidden_size, self.num_slot_type_labels, args.dropout_rate)
        if args.combine_local_context:
            self.local_context = NgramLSTM(4, config.hidden_size, args.dropout_rate)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, slot_type_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        slot_logits = self.slot_classifier(sequence_output)
        if self.args.combine_local_context:
            tmp = sequence_output.clone()[:, 1:, :]
            sequence_output = sequence_output.clone()
            sequence_output[:, 1:, :] = 0.5*(tmp + self.local_context(tmp)) 
            slot_type_logits = self.slot_type_classifier(sequence_output)
        else:
            slot_type_logits = self.slot_type_classifier(sequence_output)

        total_loss = 0

        # 2. Slot Softmax + slot type softmax
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
                    active_slot_type_logits = slot_type_logits.view(-1, self.num_slot_type_labels)[active_loss]

                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    active_slot_type_labels = slot_type_labels_ids.view(-1)[active_loss]

                    slot_loss = slot_loss_fct(active_logits, active_labels) + slot_loss_fct(active_slot_type_logits, active_slot_type_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1)) \
                        + slot_loss_fct(slot_type_logits.view(-1, self.num_slot_labels), slot_type_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss


        outputs = ((pooled_output, slot_logits, slot_type_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
