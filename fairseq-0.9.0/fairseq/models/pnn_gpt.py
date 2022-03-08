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
from fairseq.models.transformer_lm import TransformerLanguageModel

DEFAULT_MAX_TARGET_POSITIONS = 1024

@register_model('pnn_gpt')
class PNN_GPT(FairseqLanguageModel):

    def __init__(self, args, GPT_dict):
        super().__init__(GPT_dict['0'].decoder)
        self.args = args
        self.GPT_dict = GPT_dict
        
        self.Linear_dict = nn.ModuleDict()
        
        for k in range(5):
            for j in range(k):
                for i in range(self.args.decoder_layers + 1):
                    self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)] = nn.Linear(self.args.decoder_embed_dim, self.args.decoder_embed_dim)
        


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', default=2, type=int, metavar='N',
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_pnn_gpt_architecture(args)

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)
        
        GPT_dict = nn.ModuleDict()
        #for i, domain in enumerate(args.domains):
        for i in range(5):
            GPT_dict[str(i)] = TransformerLanguageModel.build_model(args, task)
        
        return cls(args, GPT_dict)

    def forward(self, src_tokens, encoder_out=None,
        incremental_state=None,
        features_only=False, cur=0, **kwargs):
        
        assert cur is not None
        past_hidden_list = []
        with torch.no_grad():
            for k in range(cur):
                hidden_sum = [0] * (self.args.decoder_layers + 1)
                for j in range(k):
                    past_hidden = past_hidden_list[j]
                    for i in range(len(past_hidden)):
                        hidden_sum[i] = hidden_sum[i] + self.Linear_dict[str(i) + '_' + str(j) + '_' + str(k)](past_hidden[i])
                if k > 0:
                    for i in range(len(hidden_sum)):
                        hidden_sum[i] = hidden_sum[i].detach()
                self.GPT_dict[str(k)].eval()
                x, extra = self.GPT_dict[str(k)](src_tokens, encoder_out=None,
        incremental_state=None,
        features_only=False, past_hidden_sum=hidden_sum, **kwargs)

                past_hidden_list.append(extra["inner_states"])
        
        k = cur
        
        hidden_sum = [0] * (self.args.decoder_layers + 1)
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
        
        x, extra = self.GPT_dict[str(k)](src_tokens, domain_labels=domain_labels, features_only=features_only, return_all_hiddens=False, classification_head_name=None, past_hidden_sum=hidden_sum, **kwargs)
        
        
        return x, extra



@register_model_architecture('pnn_gpt', 'pnn_gpt')
def base_pnn_gpt_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, 'no_tie_adaptive_proj'):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.add_bos_token = getattr(args, 'add_bos_token', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)



@register_model_architecture('pnn_gpt', 'pnn_gpt_6layer_384hidden_6head_1536ffn')
def pnn_gpt_6_384(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 384)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1536)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 6)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_pnn_gpt_architecture(args)

