# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('continual_KI')
class Continual_KI(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        self.temperature = args.temperature_distil
        self.restrict_ce_to_mask = args.restrict_ce_to_mask
        self.max_update_distil = args.max_update_distil
        self.distil_then_mlm = args.distil_then_mlm
        super().__init__(args, task)

    def forward(self, model, model_distil_from, num_updates, sample, domain=None, replay=False, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert domain != None
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        domain_labels = torch.ones_like(sample['target'], dtype=int, device='cuda') * self.args.domain_idx[domain]
        sample_size = masked_tokens.int().sum().item()
        #print(sample_size)

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits_distil_from = None
        logits_distil_s = None
        logits_distil_t = None 
        logits = None
        if self.restrict_ce_to_mask:
            logits = model(**sample['net_input'], domain_labels=domain_labels, masked_tokens=masked_tokens)[0]
            if model_distil_from:
                with torch.no_grad():
                    logits_distil_from = model_distil_from.model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        else:
            # remove calculation on special tokens.
            raise NotImplementedError
            logits_all = model(**sample['net_input'], domain_labels=domain_labels, masked_tokens=masked_tokens, features_only=True)[0]
            logits = model.output_layer(logits_all, masked_tokens=masked_tokens)
            logits_distil_s = model.output_layer(logits_all, masked_tokens=None)
            if model_distil_from:
                with torch.no_grad():
                    logits_distil_from_all = model_distil_from.model(**sample['net_input'], masked_tokens=masked_tokens, features_only=True)[0]
                    logits_distil_t = model_distil_from.model.output_layer(logits_distil_from_all, masked_tokens=None)


        targets = model.get_targets(sample, [logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        loss = 0

        loss_mlm = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        loss_distil = 0
        if model_distil_from:
            if self.restrict_ce_to_mask:
                loss_distil = (
                    F.kl_div(
                        F.log_softmax(
                            logits.view(-1, logits.size(-1)) / self.temperature,
                            dim=-1,
                            dtype=torch.float32,
                        ),
                        F.softmax(
                            logits_distil_from.view(-1, logits_distil_from.size(-1)) / self.temperature,
                            dim=-1,
                            dtype=torch.float32,
                        ),
                        reduction="sum",
                    )
                    * (self.temperature) ** 2
                )
            else:
                loss_distil = (
                    F.kl_div(
                        F.log_softmax(
                            logits_distil_s.view(-1, logits_distil_s.size(-1)) / self.temperature,
                            dim=-1,
                            dtype=torch.float32,
                        ),
                        F.softmax(
                            logits_distil_t.view(-1, logits_distil_t.size(-1)) / self.temperature,
                            dim=-1,
                            dtype=torch.float32,
                        ),
                        reduction="sum",
                    )
                    * (self.temperature) ** 2
                )

        if model_distil_from == None:
            loss = loss_mlm

        else:
            if not self.distil_then_mlm:
                distil_ratio = 1.0 - float(num_updates) / self.max_update_distil
                if distil_ratio < 0:
                    distil_ratio = 0
                loss = loss_mlm * (1.0 - distil_ratio) + loss_distil * distil_ratio
            else:
                if num_updates < self.max_update_distil:
                    loss = loss_distil
                else:
                    loss = loss_mlm

        
        
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss_mlm.data) if reduce else loss_mlm.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
