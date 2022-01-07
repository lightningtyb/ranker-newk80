import os
from re import S

import torch
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
from transformers import BartTokenizer
from transformers.models.bart.modeling_bart import BartEncoder,BartDecoder,BartClassificationHead,BartPretrainedModel, BartConfig, BartModel
from typing import Union
from transformers.file_utils import is_torch_tpu_available, WEIGHTS_NAME


class BertRanker(torch.nn.Module):

    def __init__(self, model_name_or_path):
        super().__init__()

        self.config = BertConfig.from_pretrained(model_name_or_path)
        self.bert = BertForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.bert.module if hasattr(self.bert, 'module') else self.bert
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        print('input_ids size:',input_ids.size()) #[5,512]

        batch_size = input_ids.size(0) # input_ids [12,512]
        logits  = self.bert(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids).logits
        print('logits size:',logits.size()) # [5,2]

        postive_logits = logits[:, 1] # logits[12,2]
        print('postive logits size:',postive_logits.size()) #[2]

        # print('logit size',logits.size())

        # for input, score in zip(input_ids, postive_logits):
            # print('---input ids size:', input_ids.size(), ',score:', score.item())

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size//subset_num, subset_num) # [2,6]
            pairwise_labels = torch.zeros(batch_size//subset_num, dtype=torch.long).to(logits.device)# [2]
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)
            # print('input_ids size:', input_ids.size())
            # print('pairwise_labels size :', pairwise_labels.size,'pairwise_labels:',pairwise_labels)
            #
            print('postive_logits.size:', postive_logits.size)
            # print('postive_logits:', postive_logits)
            # print('pairwise_loss:', pairwise_loss)


            return pairwise_loss
        else:

            return postive_logits


class BartEncoderRanker(BartPretrainedModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)

        self.encoder = BartEncoder(config)
        # self.bart_classification_layer = BartClassificationHead(input_dim=self.config.d_model,inner_dim=self.config.d_model,num_classes=2,pooler_dropout=self.config.classifier_dropout)
        self.lm_head = nn.Linear(self.config.d_model, 2, bias=False)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, input_mask,query_lens, query_mask, device, loss_type=None, labels=None, subset_num=2):
        batch_size = input_ids.size(0)
        encoder_outputs = self.encoder(input_ids, attention_mask=input_mask)[0]  # [4,256,1024]
        encoder_outputs = encoder_outputs[:, 0, :].squeeze()  # [4,1024]
        logits = self.lm_head(encoder_outputs)  # [4, 2]

        postive_logits = logits[:, 0].unsqueeze(-1)  # [4, 1]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size // subset_num, subset_num)  # shape '[1, 2]' is invalid for input of size 3
            pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(logits.device)
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits


class BartDecoderRanker(BartPretrainedModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.decoder = BartDecoder(config)

        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, input_mask, query_lens, query_mask, device, loss_type=None, class_labels=None, subset_num=2):
        batch_size, vocab_size = input_ids.size(0), self.config.vocab_size
        # decoder_outputs = self.decoder(decoder_input_ids=input_ids, decoder_attention_mask=input_mask)[0] # [b,m,hidden]
        decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=input_mask)[0] # [b,m,hidden]

        logits = self.lm_head(decoder_outputs)  # [bs,m, v]

        if class_labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()  # b m-1
            shift_query_mask = query_mask[..., 1:].contiguous()  # query mask [b,m] query part is 1, other token is 0
            shift_labels = torch.mul(shift_labels, shift_query_mask)  # shift_labels [b m-1] tokens are 0 except query tokens

            if loss_type == 'mse_neg':
                shift_probs = torch.nn.Softmax(dim=-1)(shift_logits)  # b m v

                all_probs = shift_probs.reshape(batch_size//subset_num, subset_num, -1, vocab_size)  # b/2 2 m v
                all_probs_clone = all_probs.clone()
                all_probs_clone[:, 1:, :, :] = 1 - all_probs[:, 1:, :, :]  # b/2 2 m v
                all_probs_clone = all_probs_clone.view(batch_size, -1, vocab_size)  # b m v
                all_probs_clone = torch.log(all_probs_clone)  # b m v

                loss_nll = torch.nn.NLLLoss(ignore_index=0, reduction='none')
                nll_loss = loss_nll(all_probs_clone.view(-1, all_probs_clone.size(-1)), shift_labels.view(-1))  # b*m
                loss = (nll_loss.reshape(batch_size, -1).sum(dim=1) / query_lens).mean()

                return loss

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
            ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # b * (m-1)

            if loss_type == 'ull':
                # unlikelihood loss
                probs = torch.exp(-ce_loss)  # b * (m-1)
                probs = probs.view(batch_size, -1)  # b , (m-1) # all 1 except query
                class_labels = class_labels.unsqueeze(1)  # b, 1
                probs = class_labels * probs + (1 - class_labels) * (1 - probs)
                log_probs = - torch.log(probs[shift_labels[:, :] > 0] + 1e-6)
                ull_loss = log_probs.mean()

                return ull_loss
            else:
                log_likelihood = ce_loss.reshape(batch_size, -1)  # b m-1 # only query has value ,other 0
                scores = log_likelihood.sum(dim=1) / query_lens  # b

                if loss_type == 'mse':
                    return scores.mean()
                else:

                    scores = scores.reshape(batch_size // subset_num, subset_num)  # pair (postive, negtive)

                    if loss_type == 'ce':
                        labels = torch.zeros(scores.size(0)).to(device)
                        loss_fct2 = torch.nn.CrossEntropyLoss(reduction='mean')
                        ranking_loss = loss_fct2(scores, labels.long())

                        return ranking_loss, scores.mean()

                    else:
                        # hinge loss only one pos one neg
                        # hinge_loss = max(0 * scores[0], 1 + scores[0] - min(scores[1:]))
                        min_neg = scores[:, 1:].min(1)[0].reshape(batch_size // subset_num) # min(scores[1:])
                        right = 1 + scores[:, 0] - min_neg # 1 + scores[0] - min(scores[1:])
                        left = 0 * scores[:, 0]
                        _hinge_loss = torch.stack((left, right), dim=-1)
                        hinge_loss = _hinge_loss.max(1)[0].mean()

                        return hinge_loss
        else:
            # inference
            input_ids_clone = input_ids.clone()
            input_ids_clone[:, :-1] = input_ids[:, 1:]  # left shift one token
            all_ids = input_ids_clone.view(-1)  # [b,m] -->[b*m]

            probs = torch.nn.LogSoftmax(dim=-1)(logits)  # [b,m,v]
            all_probs = probs.view(-1, vocab_size)[range(all_ids.size(0)), all_ids]
            all_probs = all_probs.view(batch_size, -1)  # [b,m] probabilities of all tokens

            query_mask_clone = query_mask.clone()
            query_mask_clone[:, :-1] = query_mask[:, 1:]

            query_probs = torch.mul(all_probs, query_mask_clone)  # only query probs
            score = torch.sum(query_probs, dim=1)  # [b]

            return score, logits
