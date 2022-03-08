# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    BaseFairseqModel, 
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.roberta import RobertaModel, RobertaEncoder


@register_model('pnn_roberta')
class PNN_Roberta(FairseqLanguageModel):

    def __init__(self, args, Roberta_dict):
        super().__init__(Roberta_dict['0'].decoder)
        self.args = args
        self.Roberta_dict = Roberta_dict
        
        self.Linear_dict = nn.ModuleDict()
        
        for k in range(5):
            for j in range(k):
                for i in range(self.args.encoder_layers + 1):
                    self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)] = nn.Linear(self.args.encoder_embed_dim, self.args.encoder_embed_dim)
        


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample
        
        Roberta_dict = nn.ModuleDict()
        #for i, domain in enumerate(args.domains):
        for i in range(5):
            Roberta_dict[str(i)] = RobertaModel.build_model(args, task)
        
        return cls(args, Roberta_dict)

    def forward(self, src_tokens, domain_labels=None, features_only=False, return_all_hiddens=False, classification_head_name=None, cur=0, **kwargs):
        
        assert cur is not None
        past_hidden_list = []
        with torch.no_grad():
            for k in range(cur):
                hidden_sum = [0] * (self.args.encoder_layers + 1)
                for j in range(k):
                    past_hidden = past_hidden_list[j]
                    for i in range(len(past_hidden)):
                        hidden_sum[i] = hidden_sum[i] + self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)](past_hidden[i])
                if k > 0:
                    for i in range(len(hidden_sum)):
                        hidden_sum[i] = hidden_sum[i].detach()
                self.Roberta_dict[str(k)].eval()
                x, extra = self.Roberta_dict[str(k)](src_tokens, domain_labels=domain_labels, features_only=False, return_all_hiddens=True, classification_head_name=None, past_hidden_sum=hidden_sum, **kwargs)

                past_hidden_list.append(extra["inner_states"])
        
        k = cur
        
        hidden_sum = [0] * (self.args.encoder_layers + 1)

        hidden_sum = [0] * (self.args.encoder_layers + 1)
        if k > 0:
            for i in range(len(hidden_sum)):
                try:
                    hidden_sum[i] = hidden_sum[i].detach()
                except:
                    pass
                    
        for j in range(k):
            past_hidden = past_hidden_list[j]
            for i in range(len(past_hidden)):
                hidden_sum[i] = hidden_sum[i] + self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)](past_hidden[i])
        
        
        x, extra = self.Roberta_dict[str(k)](src_tokens, domain_labels=domain_labels, features_only=features_only, return_all_hiddens=False, classification_head_name=None, past_hidden_sum=hidden_sum, **kwargs)
        
        
        return x, extra



@register_model_architecture('pnn_roberta', 'pnn_roberta')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)


@register_model_architecture('pnn_roberta', 'pnn_roberta_6layer_384hidden_6head_1536ffn')
def roberta_6_384_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 384)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1536)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 6)
    base_architecture(args)

