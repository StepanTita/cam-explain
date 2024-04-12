from typing import Any, List

import torch
import torch.nn as nn

from transformers import AutoModel

from eraserbenchmark.rationale_benchmark.models.model_utils import PaddedSequence

from space_model.model import *
from space_model.loss import *

class SpaceClassifier(nn.Module):
    """Thin wrapper around SpaceModelForSequenceClassification"""

    def __init__(self,
                 bert_dir: str,
                 pad_token_id: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 num_labels: int,
                 max_length: int = 512,
                 config: dict = None,
                 device: str = 'cuda'
                 ):
        super(SpaceClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(bert_dir)

        self.space_model = SpaceModelForSequenceClassification(
            self.base_model,
            n_embed=768, n_latent=config['n_latent'],
            n_concept_spaces=num_labels,
            l1=config['l1'],
            l2=config['l2'],
            ce_w=config['cross_entropy_weight'],
            fine_tune=True
        ).to(device)

        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these params on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
            position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id,
                                            device=target_device)
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        out = self.space_model(bert_input.data,
                        attention_mask=bert_input.mask(on=0.0, off=float('-inf'), device=target_device),
                        position_ids=positions.data)
        # assert torch.all(classes == classes) # for nans
        return out.logits
