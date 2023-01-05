# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn

import esm
from esm.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer
import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--output-file', type=str, default=None)

args = parser.parse_args()

    
log_file = open(args.output_file, 'w')
class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm.data.Alphabet):
            alphabet = esm.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout

        self._init_submodules()

    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True        
            
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result      
      
    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]
      
    def mh_sampling(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        bsz, T = tokens.size()
        
        log_file.write("Start with: ")
        log_file.flush()
            
        for seq in tokens:
                decoded_seq = ''.join([self.alphabet.all_toks[_x] for _x in seq])
                print(decoded_seq)
                log_file.write(decoded_seq + '\n')
                log_file.flush()
                
        for e in range(100):
            print('Epoch:', e)
            log_file.write("Epoch: "+ str(e) + '\n')
            log_file.flush()
            idx = torch.tensor(list(range(1, T)))
            perm_idx = torch.randperm(len(idx))
            rand_idx = idx[perm_idx] # [:2]
            for _i in tqdm(rand_idx):
                E_old = self.get_energy(tokens)
                w_0 = tokens[:, _i]
                      
                tokens_prime, w_n = self.sample_mlm(tokens, _i, tmp=args.temperature)

                q_xp_x, q_x_xp = self.get_proposal_prob(tokens, _i, w_0, w_n)
                
                E_new = self.get_energy(tokens_prime)
                
                accept_prob = (torch.exp(- E_new) * q_x_xp / torch.exp(- E_old) / q_xp_x).clamp(max=1)

                u = torch.rand(accept_prob.size()).to(accept_prob)
                
                accept_cond = (u <= accept_prob).squeeze(1)
                tokens[accept_cond] = tokens_prime[accept_cond]
                tokens = tokens.clone()
                
            # if e > 5:
            for seq in tokens:
                decoded_seq = ''.join([self.alphabet.all_toks[_x] for _x in seq])
                print(decoded_seq)
                log_file.write(decoded_seq + '\n')
                log_file.flush()
        
    def get_energy(self, tokens):
        bsz, T = tokens.size()
        energy = 0
        for _i in range(T):
            new_tokens = tokens.clone()
            new_tokens[:, _i] = self.mask_idx
            raw_logits = self.forward(tokens)['logits'].log_softmax(-1)[:, _i]
            one_energy = torch.gather(raw_logits, 1, tokens[:, _i].unsqueeze(1))
            energy += one_energy
        return energy
    
    def get_proposal_prob(self, tokens, n, w_0, w_n):
        tokens = tokens.clone()
        tokens[:, n] = self.mask_idx
        mask_prob = self.forward(tokens)['logits'].softmax(-1)[:, n]
        q_xp_x = mask_prob.gather(1, w_n.unsqueeze(1))
        q_x_xp = mask_prob.gather(1, w_0.unsqueeze(1))
        return q_xp_x, q_x_xp
            

    def sample_mlm(self, tokens, n, tmp):
        from torch.distributions import Categorical
        
        tokens = tokens.clone()
        tokens[:, n] = self.mask_idx
        logits = self.forward(tokens)['logits']
        dist = Categorical(logits=logits / tmp)
        w_n = dist.sample()[:, n]
        tokens[:, n] = w_n
        return tokens, w_n
