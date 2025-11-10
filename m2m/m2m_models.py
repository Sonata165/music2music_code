import os
import sys

sys.path.append('..')

from utils_common.utils import * 
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class M2mClsModel4PosDoubleHead(nn.Module):
    '''
    The musecoco model with 2 classification head on top, for bar-level classification of 4-bar input phrase.
    '''
    def __init__(self, pt_ckpt, head1_vocab_size=13, head2_vocab_size=10, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        3 types + 1 NA for head2_vocab_size (quality)
        '''
        super().__init__()

        m2m_lm = GPT2LMHeadModel.from_pretrained(pt_ckpt, torch_dtype=torch.bfloat16)
        m2m_config = GPT2Config.from_pretrained(pt_ckpt)
        
        self.decoder = m2m_lm.transformer

        self.proj_pos1 = nn.Linear(m2m_config.n_embd, m2m_config.n_embd)
        self.proj_pos2 = nn.Linear(m2m_config.n_embd, m2m_config.n_embd)
        self.proj_pos3 = nn.Linear(m2m_config.n_embd, m2m_config.n_embd)
        self.proj_pos4 = nn.Linear(m2m_config.n_embd, m2m_config.n_embd)

        self.cls_head_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)
        self.cls_head_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)

    def forward(self, input_ids):
        ''' Find the [EOS]'s (id=1) position, the last token of the input '''
        # If no [EOS] is found, use the last token
        target_id = 1
        bs, seq_len = input_ids.shape
        # Initialize the positions with the last index
        s_positions = torch.full((bs,), seq_len-1, dtype=torch.long)

        # Iterate over each sequence in the batch
        for i in range(bs):
            # Find the index of the first occurrence of the target_id
            target_indices = (input_ids[i] == target_id).nonzero(as_tuple=True)[0]
            if len(target_indices) > 0:
                s_positions[i] = target_indices[0]

        # s_positions = (input_ids == 2).nonzero()[:, 1]
        # if s_positions.size(0) == 0:
        #     s_positions = torch.tensor([input_ids.size(1) - 1], device=input_ids.device)
        # s_positions = torch.where(input_ids == 2)[1] # [bs, 1]

        # Original implementation
        transformer_outputs = self.decoder(
            input_ids
        )
        hidden_states = transformer_outputs['last_hidden_state']  # [bs, seq_len, dim]

        # Get the output when </s> is input
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), s_positions, :] # [bs, dim]

        h_pos1 = self.proj_pos1(last_hidden_states)
        h_pos2 = self.proj_pos2(last_hidden_states)
        h_pos3 = self.proj_pos3(last_hidden_states)
        h_pos4 = self.proj_pos4(last_hidden_states)

        res1_1, res1_2 = self.cls_head_1(h_pos1), self.cls_head_2(h_pos1) # res1_1: [bs, n_vocab1], res1_2: [bs, n_vocab2]
        res2_1, res2_2 = self.cls_head_1(h_pos2), self.cls_head_2(h_pos2)
        res3_1, res3_2 = self.cls_head_1(h_pos3), self.cls_head_2(h_pos3)
        res4_1, res4_2 = self.cls_head_1(h_pos4), self.cls_head_2(h_pos4)

        res1_out = torch.stack([res1_1, res2_1, res3_1, res4_1], dim=1) # [bs, pos, n_vocab1]
        res2_out = torch.stack([res1_2, res2_2, res3_2, res4_2], dim=1) # [bs, pos, n_vocab2]

        return res1_out, res2_out


class M2mClsModel8Head(nn.Module):
    '''
    The musecoco model with 2 classification head on top, for bar-level classification of 4-bar input phrase.
    '''
    def __init__(self, pt_ckpt, head1_vocab_size=13, head2_vocab_size=10, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        3 types + 1 NA for head2_vocab_size (quality)
        '''
        super().__init__()

        if random_init is False:
            m2m_lm = GPT2LMHeadModel.from_pretrained(pt_ckpt, torch_dtype=torch.bfloat16)
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
        else:
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
            m2m_lm = GPT2LMHeadModel(m2m_config).bfloat16()

        self.decoder = m2m_lm.transformer

        self.cls_head_1_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)
        self.cls_head_2_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)
        self.cls_head_3_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)
        self.cls_head_4_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)

        self.cls_head_1_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)
        self.cls_head_2_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)
        self.cls_head_3_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)
        self.cls_head_4_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)

    def forward(self, input_ids):
        ''' Find the [EOS]'s (id=1) position, the last token of the input '''
        # If no [EOS] is found, use the last token
        target_id = 1
        bs, seq_len = input_ids.shape

        # Initialize the positions with the last index
        s_positions = torch.full((bs,), seq_len-1, dtype=torch.long)

        # Iterate over each sequence in the batch
        for i in range(bs):
            # Find the index of the first occurrence of the target_id
            target_indices = (input_ids[i] == target_id).nonzero(as_tuple=True)[0]
            if len(target_indices) > 0:
                s_positions[i] = target_indices[0]

        # Transformer forward
        transformer_outputs = self.decoder(input_ids)
        hidden_states = transformer_outputs['last_hidden_state']  # [bs, seq_len, dim]

        # Get the output when </s> is input
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), s_positions, :] # [bs, dim]

        # Get outputs
        res1_1, res1_2 = self.cls_head_1_1(last_hidden_states), self.cls_head_1_2(last_hidden_states)
        res2_1, res2_2 = self.cls_head_2_1(last_hidden_states), self.cls_head_2_2(last_hidden_states)
        res3_1, res3_2 = self.cls_head_3_1(last_hidden_states), self.cls_head_3_2(last_hidden_states)
        res4_1, res4_2 = self.cls_head_4_1(last_hidden_states), self.cls_head_4_2(last_hidden_states)

        res1_out = torch.stack([res1_1, res2_1, res3_1, res4_1], dim=1) # [bs, pos, n_vocab1]
        res2_out = torch.stack([res1_2, res2_2, res3_2, res4_2], dim=1) # [bs, pos, n_vocab2]

        return res1_out, res2_out
    

class M2mClsModel8HeadPool(nn.Module):
    '''
    The musecoco model with 8 classification head on top, for 2-bar-level classification.
    '''
    def __init__(self, pt_ckpt, head1_vocab_size=13, head2_vocab_size=10, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        3 types + 1 NA for head2_vocab_size (quality)
        '''
        super().__init__()

        if random_init is False:
            m2m_lm = GPT2LMHeadModel.from_pretrained(pt_ckpt, torch_dtype=torch.bfloat16)
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
        else:
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
            m2m_lm = GPT2LMHeadModel(m2m_config).bfloat16()

        self.decoder = m2m_lm.transformer

        self.cls_head_1_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)
        self.cls_head_2_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)
        self.cls_head_3_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)
        self.cls_head_4_1 = nn.Linear(m2m_config.n_embd, head1_vocab_size)

        self.cls_head_1_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)
        self.cls_head_2_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)
        self.cls_head_3_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)
        self.cls_head_4_2 = nn.Linear(m2m_config.n_embd, head2_vocab_size)

    def forward(self, input_ids):
        ''' Find the [EOS]'s (id=1) position, the last token of the input '''
        # If no [EOS] is found, use the last token
        target_id = 1
        bs, seq_len = input_ids.shape

        # Initialize the positions with the last index
        s_positions = torch.full((bs,), seq_len-1, dtype=torch.long)

        # Iterate over each sequence in the batch
        for i in range(bs):
            # Find the index of the first occurrence of the target_id
            target_indices = (input_ids[i] == target_id).nonzero(as_tuple=True)[0]
            if len(target_indices) > 0:
                s_positions[i] = target_indices[0]

        # Transformer forward
        transformer_outputs = self.decoder(input_ids)
        hidden_states = transformer_outputs['last_hidden_state']  # [bs, seq_len, dim]

        # Average the hidden states of all tokens, from the first token to the s_position
        bs, seq_len, dim = hidden_states.size()
        avg_hidden_states = torch.zeros(size=(bs, dim), device=hidden_states.device)
        for i in range(bs):
            avg_hidden_states[i] = hidden_states[i, :s_positions[i]+1].mean(dim=0)

        # Get outputs
        last_hidden_states = avg_hidden_states
        res1_1, res1_2 = self.cls_head_1_1(last_hidden_states), self.cls_head_1_2(last_hidden_states)
        res2_1, res2_2 = self.cls_head_2_1(last_hidden_states), self.cls_head_2_2(last_hidden_states)
        res3_1, res3_2 = self.cls_head_3_1(last_hidden_states), self.cls_head_3_2(last_hidden_states)
        res4_1, res4_2 = self.cls_head_4_1(last_hidden_states), self.cls_head_4_2(last_hidden_states)

        res1_out = torch.stack([res1_1, res2_1, res3_1, res4_1], dim=1) # [bs, pos, n_vocab1]
        res2_out = torch.stack([res1_2, res2_2, res3_2, res4_2], dim=1) # [bs, pos, n_vocab2]

        return res1_out, res2_out


class M2mClsModelSingleHead(nn.Module):
    '''
    The musecoco model with 2 classification head on top, for bar-level classification of 4-bar input phrase.
    '''
    def __init__(self, pt_ckpt, n_class=35, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        3 types + 1 NA for head2_vocab_size (quality)
        '''
        super().__init__()

        if random_init is False:
            m2m_lm = GPT2LMHeadModel.from_pretrained(pt_ckpt, torch_dtype=torch.bfloat16)
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
        else:
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
            m2m_lm = GPT2LMHeadModel(m2m_config).bfloat16()
        
        self.decoder = m2m_lm.transformer

        self.cls_head = nn.Linear(m2m_config.n_embd, n_class)

    def forward(self, input_ids):
        ''' Find the [EOS]'s (id=1) position, the last token of the input '''
        target_id = 1 # [EOS] id
        s_positions = find_token(input_ids, target_id)

        # Transformer forward
        transformer_outputs = self.decoder(input_ids)
        hidden_states = transformer_outputs['last_hidden_state']  # [bs, seq_len, dim]

        # Get the output when </s> is input
        last_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), s_positions, :] # [bs, dim]

        # Get outputs
        out = self.cls_head(last_hidden_states)

        return out
    
class M2mClsModelSingleHeadPool(nn.Module):
    '''
    The musecoco model with 2 classification head on top, for bar-level classification of 4-bar input phrase.
    '''
    def __init__(self, pt_ckpt, n_class=35, random_init=False):
        '''
        12 note names + 1 NA for head1_vocab_size (root)
        3 types + 1 NA for head2_vocab_size (quality)
        '''
        super().__init__()

        if random_init is False:
            m2m_lm = GPT2LMHeadModel.from_pretrained(pt_ckpt, torch_dtype=torch.bfloat16)
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
        else:
            m2m_config = GPT2Config.from_pretrained(pt_ckpt)
            m2m_lm = GPT2LMHeadModel(m2m_config).bfloat16()
        
        self.decoder = m2m_lm.transformer

        self.cls_head = nn.Linear(m2m_config.n_embd, n_class)

    def forward(self, input_ids):
        ''' Find the [EOS]'s (id=1) position, the last token of the input '''
        target_id = 1 # [EOS] id
        s_positions = find_token(input_ids, target_id) # [bs, 1]

        # Transformer forward
        transformer_outputs = self.decoder(input_ids)
        hidden_states = transformer_outputs['last_hidden_state']  # [bs, seq_len, dim]

        # Average the hidden states of all tokens, from the first token to the s_position
        bs, seq_len, dim = hidden_states.size()
        avg_hidden_states = torch.zeros(size=(bs, dim), device=hidden_states.device)
        for i in range(bs):
            avg_hidden_states[i] = hidden_states[i, :s_positions[i]+1].mean(dim=0)

        # Get outputs
        out = self.cls_head(avg_hidden_states)

        return out


def find_token(input_ids, target_id=1):
    '''
    Find a specific token in the input_ids tensor.
    If the token is not found, return the last token's position.

    input_ids: [bs, seq_len], a integer tensor
    '''
    bs, seq_len = input_ids.shape

    # Initialize the positions with the last index
    s_positions = torch.full((bs,), seq_len-1, dtype=torch.long)

    # Iterate over each sequence in the batch
    for i in range(bs):
        # Find the index of the first occurrence of the target_id
        target_indices = (input_ids[i] == target_id).nonzero(as_tuple=True)[0]
        if len(target_indices) > 0:
            s_positions[i] = target_indices[0]
    
    return s_positions