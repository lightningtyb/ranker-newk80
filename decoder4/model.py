import os
from re import S

import torch
import torch.nn as nn
from torch.nn import Softmax
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer
from transformers import BartConfig, BartTokenizer, BartModel, PretrainedBartModel, BartForSequenceClassification, \
    GPT2LMHeadModel
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartClassificationHead, \
    BartPretrainedModel, BartModel, BartConfig,BartForConditionalGeneration
# from transformers.modles.gpt.modeling_gpt import GPT2LMHeadModel


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
        batch_size = input_ids.size(0)  # input_ids [12,512]
        logits = self.bert(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids).logits
        postive_logits = logits[:, 1]  # logits[12,2]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size // subset_num, subset_num)  # [2,6]
            pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(logits.device)  # [2]
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits


class RobertaRanker(torch.nn.Module):

    def __init__(self, model_name_or_path, pattern_id=0):
        super().__init__()
        self.pattern_id = pattern_id

        self.config = RobertaConfig.from_pretrained(model_name_or_path)
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.roberta.module if hasattr(self.roberta, 'module') else self.roberta
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        batch_size = input_ids.size(0)
        logits = self.roberta(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids).logits
        postive_logits = logits[:, 1]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size // subset_num, subset_num)
            pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(postive_logits.device)
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits


class T5Ranker(torch.nn.Module):
    # TODO
    def __init__(self, model_name_or_path, pattern_id=0):
        super().__init__()
        self.pattern_id = pattern_id

        self.config = RobertaConfig.from_pretrained(model_name_or_path)
        self.roberta = RobertaForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.roberta.module if hasattr(self.roberta, 'module') else self.roberta
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        batch_size = input_ids.size(0)
        logits = self.roberta(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids).logits
        postive_logits = logits[:, 1]

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size // subset_num, subset_num)
            pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(postive_logits.device)
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits


class BartEncoderModel(BartPretrainedModel):

    def __init__(self, config: BartConfig):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs


class BartEncoderRanker(torch.nn.Module):

    def __init__(self, model_name_or_path):
        super().__init__()

        self.config = BartConfig.from_pretrained(model_name_or_path)
        self.bart_encoder = BartEncoderModel.from_pretrained(model_name_or_path, config=self.config)
        self.bart_classification_layer = BartClassificationHead(input_dim=self.config.d_model,
                                                                inner_dim=self.config.d_model, num_classes=1,
                                                                pooler_dropout=self.config.classifier_dropout)
        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

        # self._init_weights(self.bart_classification_layer.dense)
        # self._init_weights(self.bart_classification_layer.out_proj)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.bart_encoder.module if hasattr(self.bart_encoder, 'module') else self.bart_encoder
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        batch_size = input_ids.size(0)
        encoder_outputs = self.bart_encoder(input_ids, attention_mask=input_mask)[0]  # [12,512,768]
        encoder_outputs = encoder_outputs[:, -1, :].squeeze()  # [bs,768]
        logits = self.bart_classification_layer(encoder_outputs)  # [bs, 2] [12,3]

        postive_logits = logits

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            postive_logits = postive_logits.reshape(batch_size // subset_num, subset_num)
            pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(logits.device)
            pairwise_loss = loss_fct(postive_logits, pairwise_labels)

            return pairwise_loss
        else:
            return postive_logits


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartDecoderModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.decoder.embed_tokens = self.shared

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return decoder_outputs


class BartDecoderRanker(torch.nn.Module):

    def __init__(self, model_name_or_path):
        super().__init__()

        self.config = BartConfig.from_pretrained(model_name_or_path)
        self.bart_decoder = BartDecoderModel.from_pretrained(model_name_or_path, config=self.config)
        # self.bart_classification_layer = BartClassificationHead(input_dim=self.config.d_model,
        #                                                         inner_dim=self.config.d_model, num_classes=1,
        #                                                         pooler_dropout=self.config.classifier_dropout)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.bart_decoder.shared.num_embeddings)))

        self.lm_head = nn.Linear(self.config.d_model, self.bart_decoder.shared.num_embeddings,bias=False) # config.vocab_size
        # self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size,bias=False) # config.vocab_size

        self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
        # self.tokenizer = self.add_special_tokens(model_name_or_path)
        # self.init_weights()
        # self.bart_decoder.resize_token_embeddings(len(self.tokenizer))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_to_save = self.bart_decoder.module if hasattr(self.bart_decoder, 'module') else self.bart_decoder
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def add_special_tokens(self,model_name_or_path):
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
        special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
        num_add_toks = tokenizer.add_special_tokens(special_tokens)
        return tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
        batch_size = input_ids.size(0) # input_ids, token_type_ids [bs,max_len]
        sep_idxs = token_type_ids.argmax(-1) # q 前面的</s>的索引 [bs] 返回每个样本q前面的</s>的索引
        decoder_outputs = self.bart_decoder(input_ids,
                                            decoder_input_ids=input_ids, decoder_attention_mask=input_mask)[0]  # [12,512,768] 0对应last_hidden_state,1对应past-key-values
        logits = self.lm_head(decoder_outputs) #[bs,max_len,vocab_len]

        _logprobs_batch = 0
        shift_logits = []
        for i in range(batch_size):
            sep_idx = sep_idxs[i]
            shift_logit = logits[i][sep_idx:-1,:] # [从q开始的位置，vocab_size]
            shift_label = input_ids[i][sep_idx+1:] # label [从q开始的位置]
            # print('logit size:',shift_logit.size(),'shift label size:',shift_label.size())
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)  # ignores padding token for loss calculation
                loss = loss_fct(shift_logit,shift_label)
                print('loss',loss)
                # _logproba = torch.nn.LogSoftmax(dim=-1)(shift_logit) # [443, 50265]
                # print('ba size:',_logproba.size())
                # _logprobs = -torch.nn.NLLLoss(reduction='none',ignore_index=self.tokenizer.pad_token_id)(_logproba, shift_label) # [443]
                # print('bs size:',_logprobs.size())
                _logprobs_batch += loss
            else:
                shift_logits.append(shift_logit)

        _logprobs_batch = _logprobs_batch / batch_size
        # _logprobs_batch = torch.sum(_logprobs_batch, dim=1)
        print('bs batch:',_logprobs_batch)
        if labels is not None:
            return _logprobs_batch
        else:
            return _logprobs_batch

        # if labels is not None:
        #     _logproba = torch.nn.LogSoftmax(dim=-1)(shift_logits.logits)
        #     _logprobs = -torch.nn.NLLLoss(reduction='none')(_logproba.transpose(1, 2), shift_label)
        #     _logprobs_batch = torch.sum(_logprobs, dim=1)
        #
        #     return _logprobs_batch
        # else:
        #     return shift_logits

        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     loss = loss_fct(shift_logits,shift_label)
        #
        # return ((loss,) + shift_logits) if loss is not None else shift_logits

        # masked_lm_loss = None
        # if labels is not None:
        #     # postive_logits = logits.reshape(batch_size // subset_num, subset_num)
        #     _logproba = torch.nn.LogSoftmax(dim=-1)(logits.logits)
        #     pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(logits.device)
        #     _logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba.transpose(1, 2), pairwise_labels)
        #     _logprobs_batch = torch.sum(_logprobs, dim=1)

        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        #
        # output = (logits,) + decoder_outputs[1:]
        # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # postive_logits = logits
        #
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss(ignore_index=-1)
        #     postive_logits = logits.reshape(batch_size // subset_num, subset_num)
        #     pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(logits.device)
        #     pairwise_loss = loss_fct(postive_logits, pairwise_labels)
        #
        #
        # return (pairwise_loss,) + output if masked_lm_loss is not None else output


        # if labels is not None:
        #     _logproba = torch.nn.LogSoftmax(dim=-1)(output.logits)
        #     _logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba.transpose(1, 2), _labels)
        #     _logprobs_batch = torch.sum(_logprobs, dim=1)
        #
        #     return _logprobs_batch
        # else:
        #     return output_dict.logits



    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past





# class BartDecoderRanker(torch.nn.Module):
#     def __init__(self, model_name_or_path):
#         super().__init__()
#         self.config = BartConfig.from_pretrained(model_name_or_path)
#         # self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
#         self.bart_decoder = BartDecoderModel.from_pretrained(model_name_or_path, config=self.config)
#         self.tokenizer = BartTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
#
#     def save(self, path):
#         os.makedirs(path, exist_ok=True)
#         model_to_save = self.bart_decoder.module if hasattr(self.bart_decoder, 'module') else self.bart_decoder
#         model_to_save.save_pretrained(path)
#         self.tokenizer.save_pretrained(path)
#
#     def get_tokenizer(self):
#         return self.tokenizer
#
#     def forward(self, input_ids, token_type_ids, input_mask, labels=None, subset_num=2):
#         batch_size = input_ids.size(0)
#         decoder_outputs = self.bart_decoder(input_ids, attention_mask=input_mask,
#                                             decoder_input_ids=input_ids, decoder_attention_mask=input_mask)[0]  # [12,512,768]
#         # decoder_outputs = self.decoder(input_ids, attention_mask=input_mask)[0]  # [12,512,768]
#         output = decoder_outputs[:, -1, :].squeeze()  # [bs,768]
#         # logits = self.bart_classification_layer(decoder_outputs)  # [bs, 2] [12,3]
#
#         # postive_logits = logits
#         #
#         # if labels is not None:
#         #     loss_fct = CrossEntropyLoss(ignore_index=-1)
#         #     postive_logits = postive_logits.reshape(batch_size // subset_num, subset_num)
#         #     pairwise_labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(logits.device)
#         #     pairwise_loss = loss_fct(postive_logits, pairwise_labels)
#         #
#         #     return pairwise_loss
#         # else:
#         #     return postive_logits
#
#         if labels is not None:
#             _logproba = torch.nn.LogSoftmax(dim=-1)(output.logits)
#
#             _labels = torch.zeros(batch_size // subset_num, dtype=torch.long).to(output.device)
#
#             _logprobs = -torch.nn.NLLLoss(ignore_index=0, reduction='none')(_logproba.transpose(1, 2), _labels)
#             _logprobs_batch = torch.sum(_logprobs, dim=1)
#
#             return _logprobs_batch
#         else:
#             return output.logits