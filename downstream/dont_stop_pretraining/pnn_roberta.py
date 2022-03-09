import argparse
import pathlib
import os
import fairseq
import torch
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.pnn_roberta import PNN_Roberta
from fairseq.tasks.continual_KI import Continual_KI
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version
import torch.nn as nn

from transformers import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification, PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.utils import logging
from allennlp.common import Registrable
from allennlp.models.model import Model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from overrides import overrides


class PNNRobertaConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Examples::

        >>> from transformers import RobertaConfig, RobertaModel

        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()

        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "pnn_roberta"

    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, num=0, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.num = num

class PNNRoberta(RobertaPreTrainedModel):
    base_model_prefix = "pnn_roberta"
    def __init__(self, config):
        super().__init__(config)

        self.Roberta_list = nn.ModuleList()
        for i in range(self.config.num):
            self.Roberta_list.append(RobertaModel(config))
        self.Linear_dict = nn.ModuleDict()
        self.cur = self.config.num - 1

        for k in range(self.config.num):
            for j in range(k):
                for i in range(self.config.num_hidden_layers + 1):
                    self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)] = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        
        past_hidden_list = []
        for k in range(self.cur):
            #print(k)
            hidden_sum = [0] * (self.config.num_hidden_layers + 1)
            with torch.no_grad():
                for j in range(k):
                    past_hidden = past_hidden_list[j]
                    for i in range(len(past_hidden)):
                        hidden_sum[i] = hidden_sum[i] + self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)](past_hidden[i])
                
                if k > 0:
                    for i in range(len(hidden_sum)):
                        hidden_sum[i] = hidden_sum[i].detach()

                self.Roberta_list[k].eval()
                outputs = self.Roberta_list[k](input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                position_ids=position_ids,
                                                head_mask=head_mask,
                                                inputs_embeds=inputs_embeds,
                                                encoder_hidden_states=encoder_hidden_states,
                                                encoder_attention_mask=encoder_attention_mask,
                                                output_attentions=output_attentions,
                                                output_hidden_states=True,
                                                return_dict=return_dict,
                                                past_hidden_sum=hidden_sum)
                #print(len(outputs.hidden_states))
                #print(outputs.hidden_states[0].size())
                past_hidden_list.append(outputs.hidden_states)
        
        k = self.cur
        #self.Roberta_list[k].train()
        hidden_sum = [0] * (self.config.num_hidden_layers + 1)
        '''
        if k > 0:
            for i in range(len(hidden_sum)):
                try:
                    hidden_sum[i] = hidden_sum[i].detach()
                except:
                    pass
        '''
        for j in range(k):
            past_hidden = past_hidden_list[j]
            for i in range(len(past_hidden)):
                hidden_sum[i] = hidden_sum[i] + self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)](past_hidden[i])
        
        outputs = self.Roberta_list[k](input_ids,
                                                attention_mask=attention_mask,
                                                token_type_ids=token_type_ids,
                                                position_ids=position_ids,
                                                head_mask=head_mask,
                                                inputs_embeds=inputs_embeds,
                                                encoder_hidden_states=encoder_hidden_states,
                                                encoder_attention_mask=encoder_attention_mask,
                                                output_attentions=output_attentions,
                                                output_hidden_states=output_hidden_states,
                                                return_dict=return_dict,
                                                past_hidden_sum=hidden_sum)
        
        return outputs

@TokenEmbedder.register('pnn_roberta')
class PNNRobertaEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from ``pytorch-transformers`` as a ``TokenEmbedder``.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.transformer_model = PNNRoberta.from_pretrained(model_name)
        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.transformer_model.config.hidden_size

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:  # type: ignore

        return self.transformer_model(token_ids)[0]