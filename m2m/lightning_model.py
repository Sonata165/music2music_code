import os
import sys

sys.path.append('..')

import torch
import torch.nn as nn
import lightning as L
from torch import optim
import transformers
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from utils_common.utils import jpath, read_json
from utils_midi import remi_utils
from m2m.evaluate import Metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from m2m_models import *
import math
# from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
# from peft import get_peft_model, LoraConfig, TaskType


def load_lit_model(tokenizer, config, lit_ckpt=None):
    if lit_ckpt is None:
        out_dir = jpath(config['result_root'], config['out_dir'])
        latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
    else:
        ckpt_fp = lit_ckpt
    pt_ckpt = config['pt_ckpt']
    lit_model_class = eval(config['lit_model_class'])
    l_model = lit_model_class.load_from_checkpoint(ckpt_fp, pt_ckpt=pt_ckpt, tokenizer=tokenizer, infer=True)
    l_model.config = config
    return l_model


def get_lit_model(model_fp, tokenizer, config):
    lit_model_class = config['lit_model_class']
    print(lit_model_class)
    lit_model_class = eval(lit_model_class)
    lit_model = lit_model_class(
        model_fp,
        tokenizer=tokenizer, 
        hparams=config
    )
    return lit_model


class LitM2mLM(L.LightningModule):
    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()
        if infer is False and hparams.get('random_init', False) is False: 
            # Training, init from pretrained model
            self.model = GPT2LMHeadModel.from_pretrained(pt_ckpt, torch_dtype=torch.bfloat16)
        else:
            # Inference, init from random
            # Or training with random init
            config = AutoConfig.from_pretrained(pt_ckpt)
            self.model = GPT2LMHeadModel(config).bfloat16()

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

        self.test_results = {}

        if self.model.config.vocab_size != len(self.tk):
            self.model.resize_token_embeddings(len(self.tk))


    def training_step(self, batch, batch_idx):
        # Tokenize the batch
        self.tk.padding_side = 'right'
        inp_seqs = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_len'],
        )['input_ids'].cuda()

        ''' Target sequence loss '''
        if self.config['loss_type'] == 'tgt_seq':
            # Craft label: find the [SEP] token's position in each sample, and mask the tokens before [SEP]
            sep_idx = find_token(inp_seqs, self.tk.sep_token_id)
            labels_unshifted = inp_seqs.clone()
            for i in range(inp_seqs.shape[0]):
                labels_unshifted[i, :sep_idx[i]+1] = -100

            # Mask the pad tokens
            for i in range(inp_seqs.shape[0]):
                for j in range(sep_idx[i]+1, inp_seqs.shape[1]):
                    if inp_seqs[i, j] == self.tk.pad_token_id:
                        labels_unshifted[i, j] = -100

            loss = self.model(inp_seqs, labels=labels_unshifted).loss

        # ''' Full sequence loss '''
        elif self.config['loss_type'] == 'full_seq':
            loss = self.model(inp_seqs, labels=inp_seqs).loss

        # Logging to TensorBoard (if installed) by default
        sch = self.lr_schedulers()
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # LR scheduler update
        if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        bs = len(batch)
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        ''' Target sequence loss '''
        # Craft label: find the [SEP] token's position in each sample, and mask the tokens before [SEP]
        sep_idx = find_token(inp_seqs, self.tk.sep_token_id) # a more robust way
        labels_unshifted = inp_seqs.clone()
        for i in range(inp_seqs.shape[0]):
            labels_unshifted[i, :sep_idx[i]+1] = -100

        # Mask the pad tokens
        for i in range(inp_seqs.shape[0]):
            for j in range(sep_idx[i]+1, inp_seqs.shape[1]):
                if inp_seqs[i, j] == self.tk.pad_token_id:
                    labels_unshifted[i, j] = -100
        
        loss = self.model(inp_seqs, labels=labels_unshifted).loss

        # Logging to TensorBoard (if installed) by default
        self.log("valid_loss", loss, batch_size=bs)

        # Generation in validation
        valid_tot_steps = len(self.trainer.val_dataloaders)
        valid_interval = valid_tot_steps // self.config['val_gen_n_samples']
        # valid_interval = 1  # for debugging
        if batch_idx % valid_interval == 0:
            metric = Metric()

            generate_kwargs = {
                'max_length': self.config['max_len'],
                'use_cache': True, 
                'do_sample': True, # Strategy: greedy sampling
                'bad_words_ids': [[self.tk.pad_token_id]],
            }
            inp_sep_idx = sep_idx[0]
            inp_seq_gen = inp_seqs[0, :][:inp_sep_idx+1]
            attn_mask = attn_masks[0, :][:inp_sep_idx+1]
            inp_len = inp_seq_gen.shape[0]

            gen_out = self.model.generate(
                inp_seq_gen.unsqueeze(0),
                pad_token_id=self.tk.pad_token_id,
                attention_mask=attn_mask.unsqueeze(0),
                **generate_kwargs
            )[0][inp_len:]

            # Get detokenized output and input sequence
            out_str = self.tk.decode(gen_out, skip_special_tokens=True)
            out_seq = out_str.strip().split(' ')
            inp_str = self.tk.decode(inp_seq_gen, skip_special_tokens=False)
            inp_seq = inp_str.strip().split(' ')

            # Get the target sequence
            tgt_seq_ids = inp_seqs[0, :][inp_sep_idx+1:]
            tgt_seq = self.tk.decode(tgt_seq_ids, skip_special_tokens=True).strip().split(' ')

            scores = self.calculate_metrics(inp_seq, tgt_seq, out_seq)
            for k, v in scores.items():
                metric.update(k, v)

            scores = metric.average()

            for score in scores:
                self.log(f"valid_{score}", scores[score], batch_size=bs)

        return loss
    
    def validation_step_new(self, batch, batch_idx):
        bs = len(batch)

        self.tk.padding_side = 'right'
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        ''' Target sequence loss '''
        # Craft label: find the [SEP] token's position in each sample, and mask the tokens before [SEP]
        sep_idx = find_token(inp_seqs, self.tk.sep_token_id) # a more robust way
        labels_unshifted = inp_seqs.clone()
        for i in range(inp_seqs.shape[0]):
            labels_unshifted[i, :sep_idx[i]+1] = -100

        # Mask the pad tokens
        for i in range(inp_seqs.shape[0]):
            for j in range(sep_idx[i]+1, inp_seqs.shape[1]):
                if inp_seqs[i, j] == self.tk.pad_token_id:
                    labels_unshifted[i, j] = -100
        
        loss = self.model(inp_seqs, labels=labels_unshifted).loss

        # Logging to TensorBoard (if installed) by default
        self.log("valid_loss", loss, batch_size=bs)


        ''' Generation '''
        
        # Prepare input sequences for generation
        inputs = []
        targets = []
        for i in range(bs):
            tot_seq = batch[i].strip().split(' ')
            inp_seq = tot_seq[:sep_idx[i]+1]
            tgt_seq = tot_seq[sep_idx[i]+1:]
            if tgt_seq[-1] == '[EOS]':
                tgt_seq = tgt_seq[:-1]
            inputs.append(' '.join(inp_seq))
            targets.append(' '.join(tgt_seq))

        # Tokenize inputs, left padding for batch generation
        self.tk.padding_side = 'left'
        tokenized = self.tk(
            inputs, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len']-1,
        )
        input_ids = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        generate_kwargs = {
            'max_length': self.config['max_len'],
            'use_cache': True, 
            'bad_words_ids': [[self.tk.pad_token_id]],

            # Strategy: greedy decode
            'do_sample': False, 
        }

        inp_len = input_ids.shape[1]

        gen_out = self.model.generate(
            input_ids,
            pad_token_id=self.tk.pad_token_id,
            attention_mask=attn_masks,
            **generate_kwargs
        )[:, inp_len:]

        # Get detokenized output and input sequence
        inp_strs = inputs
        tgt_strs = targets
        out_strs = self.tk.batch_decode(gen_out, skip_special_tokens=True)
        
        metric = Metric()
        for inp_str, tgt_str, out_str in zip(inp_strs, tgt_strs, out_strs):
            inp_seq = inp_str.strip().split(' ')
            tgt_seq = tgt_str.strip().split(' ')
            out_seq = out_str.strip().split(' ')

            scores = self.calculate_metrics(inp_seq, tgt_seq, out_seq)
            for k, v in scores.items():
                metric.update(k, v)

        for k in metric.metrics:
            if k not in self.test_results:
                self.test_results[k] = []
            self.test_results[k].extend(metric.metrics[k])

        scores = metric.average()

        for score in scores:
            self.log(f"valid_{score}", scores[score], batch_size=bs)

        return loss

    def test_step_bak(self, batch, batch_idx):
        bs = len(batch)
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        ''' Target sequence loss '''
        # Craft label: find the [SEP] token's position in each sample, and mask the tokens before [SEP]
        sep_idx = find_token(inp_seqs, self.tk.sep_token_id) # a more robust way
        # sep_idx = (inp_seqs == self.tk.convert_tokens_to_ids('[SEP]')).nonzero(as_tuple=True)[1]
        labels_unshifted = inp_seqs.clone()
        for i in range(inp_seqs.shape[0]):
            labels_unshifted[i, :sep_idx[i]+1] = -100
        loss = self.model(inp_seqs, labels=labels_unshifted).loss

        # Logging to TensorBoard (if installed) by default
        
        ppl = math.exp(loss)
        self.log("test_ppl", ppl, batch_size=bs)

        # Generation in validation
        valid_tot_steps = len(self.trainer.test_dataloaders)
        valid_interval = valid_tot_steps // self.config['val_gen_n_samples']
        # valid_interval = 1  # for debugging
        if batch_idx % valid_interval == 0:
            metric = Metric()

            generate_kwargs = {
                'max_length': self.config['max_len'],
                'use_cache': True, 
                'do_sample': True, # Strategy: greedy sampling
                'bad_words_ids': [[self.tk.pad_token_id]],
            }
            inp_sep_idx = sep_idx[0]
            inp_seq_gen = inp_seqs[0, :][:inp_sep_idx+1]
            attn_mask = attn_masks[0, :][:inp_sep_idx+1]
            inp_len = inp_seq_gen.shape[0]

            gen_out = self.model.generate(
                inp_seq_gen.unsqueeze(0),
                pad_token_id=self.tk.pad_token_id,
                attention_mask=attn_mask.unsqueeze(0),
                **generate_kwargs
            )[0][inp_len:]

            # Get detokenized output and input sequence
            out_str = self.tk.decode(gen_out, skip_special_tokens=True)
            out_seq = out_str.strip().split(' ')
            inp_str = self.tk.decode(inp_seq_gen, skip_special_tokens=False)
            inp_seq = inp_str.strip().split(' ')

            # Get the target sequence
            tgt_seq_ids = inp_seqs[0, :][inp_sep_idx+1:]
            tgt_seq = self.tk.decode(tgt_seq_ids, skip_special_tokens=True).strip().split(' ')

            # Get target insts
            inst_start_idx = inp_seq.index('[INST]') + 1
            # inst_end_idx = inp_seq.index('[SEP]') # for 4-bar inst and voice control model
            if '[MELODY]' not in inp_seq:
                inst_end_idx = inp_seq.index('[PITCH]') if '[PITCH]' in inp_seq else inp_seq.index('[SEP]')  # for 4-bar model
            else:
                inst_end_idx = inp_seq.index('[MELODY]')
            tgt_insts = inp_seq[inst_start_idx:inst_end_idx]

            # Get output insts
            out_insts = remi_utils.from_remi_get_inst_and_voice(out_seq)
            
            # Calculate inst iou
            inst_iou = metric.calculate_inst_iou_from_inst(out_insts, tgt_insts)
            self.log('test_inst_iou', inst_iou, batch_size=bs)

            # Calculate voice wer
            voice_wer = metric.calculate_wer(out_insts, tgt_insts)
            self.log('test_voice_wer', voice_wer, batch_size=bs)

            # # Get bar count
            # bar_count = out_seq.count('b-1')
            # self.log('bar_count', bar_count, batch_size=bs)

            # Groove similarity
            pos_wer, pos_sor = metric.calculate_groove_wer_sor(out_seq, tgt_seq)
            self.log('test_pos_wer', pos_wer, batch_size=bs)
            # self.log('pos_sor', pos_sor, batch_size=bs)

            # Melody recall
            melody_recall = metric.calculate_melody_recall(out_seq, tgt_seq)
            self.log('test_melody_recall', melody_recall, batch_size=bs)

            # Chroma similarity
            chroma_iou = metric.calculate_chroma_iou(out_seq, tgt_seq)
            self.log('test_chroma_iou', chroma_iou, batch_size=bs)

            # Inter-track diversity
            div_mean, div_std = metric.calculate_inter_track_div(out_seq)
            self.log('test_div_mean', div_mean, batch_size=bs)
            self.log('test_div_std', div_std, batch_size=bs)

        return loss

    def test_step(self, batch, batch_idx):
        

        bs = len(batch)
        
        # Perplexity need right-side padding
        self.tk.padding_side = 'right'
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        ''' Target sequence loss '''
        # Craft label: find the [SEP] token's position in each sample, and mask the tokens before [SEP]
        sep_idx = find_token(inp_seqs, self.tk.sep_token_id) # a more robust way
        # sep_idx = (inp_seqs == self.tk.convert_tokens_to_ids('[SEP]')).nonzero(as_tuple=True)[1]
        labels_unshifted = inp_seqs.clone()
        for i in range(inp_seqs.shape[0]): # input and left padding will be filled with -100
            labels_unshifted[i, :sep_idx[i]+1] = -100
        # Mask the pad tokens
        for i in range(inp_seqs.shape[0]):
            for j in range(sep_idx[i]+1, inp_seqs.shape[1]):
                if inp_seqs[i, j] == self.tk.pad_token_id:
                    labels_unshifted[i, j] = -100

        loss = self.model(inp_seqs, labels=labels_unshifted, attention_mask=attn_masks).loss
        ppl = math.exp(loss)
        self.log("test_ppl", ppl, batch_size=bs)


        ''' Generation '''
        
        # Prepare input sequences for generation
        inputs = []
        targets = []
        for i in range(bs):
            tot_seq = batch[i].strip().split(' ')
            inp_seq = tot_seq[:sep_idx[i]+1]
            tgt_seq = tot_seq[sep_idx[i]+1:]
            if tgt_seq[-1] == '[EOS]':
                tgt_seq = tgt_seq[:-1]
            inputs.append(' '.join(inp_seq))
            targets.append(' '.join(tgt_seq))

        # Tokenize inputs, left padding for batch generation
        self.tk.padding_side = 'left'
        tokenized = self.tk(
            inputs, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len']-1,
        )
        input_ids = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        generate_kwargs = {
            'max_length': self.config['max_len'],
            'use_cache': True, 
            'bad_words_ids': [[self.tk.pad_token_id]],

            # Strategy: greedy decode
            'do_sample': False, 
        }

        inp_len = input_ids.shape[1]

        gen_out = self.model.generate(
            input_ids,
            pad_token_id=self.tk.pad_token_id,
            attention_mask=attn_masks,
            **generate_kwargs
        )[:, inp_len:]

        # Get detokenized output and input sequence
        inp_strs = inputs
        tgt_strs = targets
        out_strs = self.tk.batch_decode(gen_out, skip_special_tokens=True)
        
        metric = Metric()
        for inp_str, tgt_str, out_str in zip(inp_strs, tgt_strs, out_strs):
            inp_seq = inp_str.strip().split(' ')
            tgt_seq = tgt_str.strip().split(' ')
            out_seq = out_str.strip().split(' ')

            scores = self.calculate_metrics(inp_seq, tgt_seq, out_seq)
            for k, v in scores.items():
                metric.update(k, v)

        for k in metric.metrics:
            if k not in self.test_results:
                self.test_results[k] = []
            self.test_results[k].extend(metric.metrics[k])

        scores = metric.average()

        for score in scores:
            self.log(f"test_{score}", scores[score], batch_size=bs)

        return loss

    def on_test_epoch_end(self):
        # Get logging dir
        out_dir = jpath(self.config['result_root'], self.config['out_dir'])
        latest_version_dir, ckpt_fp = get_latest_checkpoint(out_dir)
        dirof = os.path.dirname
        log_dir = dirof(dirof(ckpt_fp))
        print('Logging dir: {}'.format(log_dir))

        # Save sample-wise metrics.
        save_dir = log_dir
        save_fp = jpath(save_dir, '{}.json'.format(self.config['model_name']))
        save_json(self.test_results, save_fp)

    def calculate_metrics(self, inp_seq, tgt_seq, out_seq) -> dict:
        '''
        Calculate the metrics for a single sample
        '''
        metric = Metric()
        ret = {}

        # Get target and output insts
        tgt_insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        out_insts = remi_utils.from_remi_get_inst_and_voice(out_seq)
        
        # Calculate inst iou
        inst_iou = metric.calculate_inst_iou_from_inst(out_insts, tgt_insts)
        ret['inst_iou'] = inst_iou

        # Calculate voice wer
        voice_wer = metric.calculate_wer(out_insts, tgt_insts)
        ret['voice_wer'] = voice_wer

        # Note F1 (16th note quantized)
        note_f1 = metric.calculate_note_f1_q16(out_seq, tgt_seq)
        ret['note_f1'] = note_f1

        # Note_i F1 (16th note quantized)
        note_i_f1 = metric.calculate_note_i_f1_q16(out_seq, tgt_seq)
        ret['note_i_f1'] = note_i_f1

        # Melody F1 (16th note quantized)
        
        melody_f1 = metric.calculate_melody_f1_q16(out_seq, tgt_seq)
        ret['melody_f1'] = melody_f1

        # Groove similarity
        pos_wer, pos_sor = metric.calculate_groove_wer_sor(out_seq, tgt_seq)
        ret['pos_wer'] = pos_wer

        # Melody recall
        melody_recall = metric.calculate_melody_recall(out_seq, tgt_seq)
        ret['melody_recall'] = melody_recall

        return ret

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        if self.config['lr_scheduler'] == 'none':
            ret = {"optimizer": optimizer}

        elif self.config['lr_scheduler'] == 'linear':
            # Linear scheduler
            max_steps = self.num_training_steps()
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=max_steps,
            )
            ret = {"optimizer": optimizer, "lr_scheduler": scheduler},
        
        elif self.config['lr_scheduler'] == 'anneal':
            # Annealing
            scheduler = ReduceLROnPlateauPatch(
                optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['lr_anneal_patience'],
                verbose=True
            )

            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid_loss",
                },
            }
        
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)
    
    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        # LR anneal update
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_loss"])


class LitUnconditionalLM(LitM2mLM):
    def training_step(self, batch, batch_idx):
        bs = len(batch)
        split = 'train'
        
        # Tokenize the batch
        self.tk.padding_side = 'left'
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_mask = tokenized['attention_mask'].cuda()

        # ''' Full sequence loss '''
        labels_unshifted = inp_seqs.clone()
        labels_unshifted[attn_mask == 0] = -100
        loss = self.model(inp_seqs, labels=labels_unshifted, attention_mask=attn_mask).loss

        # Logging to TensorBoard (if installed) by default
        sch = self.lr_schedulers()
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # LR scheduler update
        if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step()

        ''' Metrics '''
        with torch.no_grad():
            # Token-level perplexity
            ppl_tok = math.exp(loss)

            # Get the total #notes and #tokens in the batch
            n_notes_of_samples = [sample.count('p') for sample in batch]
            n_notes = sum(n_notes_of_samples)
            n_tokens_of_samples = [sample.count(' ')-1 for sample in batch]
            n_tokens = sum(n_tokens_of_samples)

            # Note-level perplexity
            avg_tokens_per_note = n_tokens / n_notes
            ppl_note = ppl_tok ** avg_tokens_per_note

            self.log(f"{split}_ppl_tok", ppl_tok, batch_size=bs)
            self.log(f"{split}_ppl_note", ppl_note, batch_size=bs)
            self.log(f"{split}_tokens_per_note", avg_tokens_per_note, batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        bs = len(batch)
        split = 'valid'

        self.tk.padding_side = 'left'
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_mask = tokenized['attention_mask'].cuda()

        # ''' Full sequence loss '''
        labels_unshifted = inp_seqs.clone()
        labels_unshifted[attn_mask == 0] = -100
        loss = self.model(inp_seqs, labels=labels_unshifted, attention_mask=attn_mask).loss

        # Logging to TensorBoard (if installed) by default
        self.log("valid_loss", loss, batch_size=bs)

        ''' Metrics '''
        # Token-level perplexity
        ppl_tok = math.exp(loss)

        # Get the total #notes and #tokens in the batch
        n_notes_of_samples = [sample.count('p') for sample in batch]
        n_notes = sum(n_notes_of_samples)
        n_tokens_of_samples = [sample.count(' ')-1 for sample in batch]
        n_tokens = sum(n_tokens_of_samples)

        # Note-level perplexity
        avg_tokens_per_note = n_tokens / n_notes
        ppl_note = ppl_tok ** avg_tokens_per_note

        self.log(f"{split}_ppl_tok", ppl_tok, batch_size=bs)
        self.log(f"{split}_ppl_note", ppl_note, batch_size=bs)
        self.log(f"{split}_tokens_per_note", avg_tokens_per_note, batch_size=bs)

    def test_step(self, batch, batch_idx):
        bs = len(batch)
        split = 'test'

        self.tk.padding_side = 'left'
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_mask = tokenized['attention_mask'].cuda()

        # ''' Full sequence loss '''
        labels_unshifted = inp_seqs.clone()
        labels_unshifted[attn_mask == 0] = -100
        loss = self.model(inp_seqs, labels=labels_unshifted, attention_mask=attn_mask).loss

        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss, batch_size=bs)

        ''' Metrics '''
        # Token-level perplexity
        ppl_tok = math.exp(loss)

        # Get the total #notes and #tokens in the batch
        n_notes_of_samples = [sample.count('p') for sample in batch]
        n_notes = sum(n_notes_of_samples)
        n_tokens_of_samples = [sample.count(' ')-1 for sample in batch]
        n_tokens = sum(n_tokens_of_samples)

        # Note-level perplexity
        avg_tokens_per_note = n_tokens / n_notes
        ppl_note = ppl_tok ** avg_tokens_per_note

        self.log(f"{split}_ppl_tok", ppl_tok, batch_size=bs)
        self.log(f"{split}_ppl_note", ppl_note, batch_size=bs)
        self.log(f"{split}_tokens_per_note", avg_tokens_per_note, batch_size=bs)


class LitM2mExpander(LitM2mLM):
    def calculate_metrics(self, inp_seq, tgt_seq, out_seq) -> dict:
        '''
        Calculate the metrics for a single sample
        '''
        metric = Metric()
        ret = {}

        # Get target and output insts
        tgt_insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        out_insts = remi_utils.from_remi_get_inst_and_voice(out_seq)
        
        # Calculate inst iou
        inst_iou = metric.calculate_inst_iou_from_inst(out_insts, tgt_insts)
        ret['inst_iou'] = inst_iou

        # Calculate voice wer
        voice_wer = metric.calculate_wer(out_insts, tgt_insts)
        ret['voice_wer'] = voice_wer

        # Melody recall
        melody_recall = metric.calculate_melody_recall(out_seq, tgt_seq)
        ret['melody_recall'] = melody_recall

        # Chroma similarity
        chroma_iou = metric.calculate_chroma_iou(out_seq, tgt_seq)
        ret['chroma_iou'] = chroma_iou

        # Chord accuracy
        tgt_chord_seq = remi_utils.from_remi_get_chord_seq(tgt_seq)
        out_chord_seq = remi_utils.from_remi_get_chord_seq(out_seq)
        chord_root_acc, chord_type_acc = metric.calculate_chord_acc_from_chord(out_chord_seq, tgt_chord_seq)
        ret['chord_root_acc_4'] = chord_root_acc
        ret['chord_type_acc_4'] = chord_type_acc
        chord_root_acc, chord_type_acc = metric.calculate_chord_acc_from_chord_two_per_bar(out_chord_seq, tgt_chord_seq)
        ret['chord_root_acc_2'] = chord_root_acc
        ret['chord_type_acc_2'] = chord_type_acc

        # Inter-track diversity
        div_mean, div_std = metric.calculate_inter_track_div(out_seq)
        ret['div_mean'] = div_mean
        ret['div_std'] = div_std

        return ret


class LitM2mDrum(LitM2mLM):
    '''
    Lightning module for drum arrangement
    '''
    def validation_step(self, batch, batch_idx):
        bs = len(batch)
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        ''' Target sequence loss '''
        # Craft label: find the [SEP] token's position in each sample, and mask the tokens before [SEP]
        sep_idx = find_token(inp_seqs, self.tk.sep_token_id) # a more robust way
        labels_unshifted = inp_seqs.clone()
        for i in range(inp_seqs.shape[0]):
            labels_unshifted[i, :sep_idx[i ]+1] = -100

        # Mask the pad tokens
        for i in range(inp_seqs.shape[0]):
            for j in range(sep_idx[i]+1, inp_seqs.shape[1]):
                if inp_seqs[i, j] == self.tk.pad_token_id:
                    labels_unshifted[i, j] = -100

        loss = self.model(inp_seqs, labels=labels_unshifted).loss

        # In very rare cases, [SEP] token not found will cause entire label sequence being masked, resulting in nan loss
        if torch.isnan(loss):
            loss = 0

        # Generation in validation
        valid_tot_steps = len(self.trainer.val_dataloaders)
        valid_interval = valid_tot_steps // self.config['val_gen_n_samples']
        
        # valid_interval = 1  # for debugging
        
        if batch_idx % valid_interval == 0:
            metric = Metric()

            generate_kwargs = {
                'max_length': self.config['max_len'],
                'use_cache': True, 
                'do_sample': True, # Strategy: greedy sampling
                'bad_words_ids': [[self.tk.pad_token_id]],
            }
            inp_sep_idx = sep_idx[0]
            inp_seq_gen = inp_seqs[0, :][:inp_sep_idx+1]
            attn_mask = attn_masks[0, :][:inp_sep_idx+1]
            inp_len = inp_seq_gen.shape[0]

            gen_out = self.model.generate(
                inp_seq_gen.unsqueeze(0),
                pad_token_id=self.tk.pad_token_id,
                attention_mask=attn_mask.unsqueeze(0),
                **generate_kwargs
            )[0][inp_len:]

            # Get detokenized output and input sequence
            out_str = self.tk.decode(gen_out, skip_special_tokens=True)
            out_seq = out_str.strip().split(' ')
            inp_str = self.tk.decode(inp_seq_gen, skip_special_tokens=False)
            inp_seq = inp_str.strip().split(' ')

            # Get the target sequence
            tgt_seq_ids = inp_seqs[0, :][inp_sep_idx+1:]
            tgt_seq = self.tk.decode(tgt_seq_ids, skip_special_tokens=True).strip().split(' ')

            # Get target insts
            inst_start_idx = inp_seq.index('[INST]') + 1
            # inst_end_idx = inp_seq.index('[SEP]') # for 4-bar inst and voice control model
            if '[MELODY]' not in inp_seq:
                inst_end_idx = inp_seq.index('[PITCH]') if '[PITCH]' in inp_seq else inp_seq.index('[SEP]')  # for 4-bar model
            else:
                inst_end_idx = inp_seq.index('[MELODY]')
            tgt_insts = inp_seq[inst_start_idx:inst_end_idx]

            # Get output insts
            out_insts = remi_utils.from_remi_get_inst_and_voice(out_seq)
            
            # Calculate inst iou
            inst_iou = metric.calculate_inst_iou_from_inst(out_insts, tgt_insts)
            self.log('inst_iou', inst_iou, batch_size=bs)

            # Get bar count
            bar_count = out_seq.count('b-1')
            self.log('bar_count', bar_count, batch_size=bs)

            # Groove similarity
            pos_wer, pos_sor = metric.calculate_groove_wer_sor_mbar(out_seq, tgt_seq)
            self.log('pos_wer', pos_wer, batch_size=bs)
            self.log('pos_sor', pos_sor, batch_size=bs)

            # Drum WER and SOR
            out_pitch = remi_utils.from_remi_mbar_get_pitch_seq_of_track(out_seq, 'i-128')
            tgt_pitch = remi_utils.from_remi_mbar_get_pitch_seq_of_track(tgt_seq, 'i-128')
            drum_wer = metric.calculate_wer(out_pitch, tgt_pitch)
            self.log('drum_wer', drum_wer, batch_size=bs)
            drum_sor = metric.calculate_sor(out_pitch, tgt_pitch)
            self.log('drum_sor', drum_sor, batch_size=bs)


        # Logging to TensorBoard (if installed) by default
        self.log("valid_loss", loss, batch_size=bs)

        return loss

    def calculate_metrics(self, inp_seq, tgt_seq, out_seq) -> dict:
        '''
        Calculate the metrics for a single sample
        '''
        metric = Metric()
        ret = {}

        # Get target and output insts
        tgt_insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        out_insts = remi_utils.from_remi_get_inst_and_voice(out_seq)

        # Calculate inst iou
        inst_iou = metric.calculate_inst_iou_from_inst(out_insts, tgt_insts)
        ret['inst_iou'] = inst_iou

        # Bar count
        bar_count = out_seq.count('b-1')
        ret['bar_count'] = bar_count

        # Groove similarity
        pos_wer, pos_sor = metric.calculate_groove_wer_sor_mbar(out_seq, tgt_seq)
        ret['pos_wer'] = pos_wer

        # Drum WER and SOR
        out_pitch = remi_utils.from_remi_mbar_get_pitch_seq_of_track(out_seq, 'i-128')
        tgt_pitch = remi_utils.from_remi_mbar_get_pitch_seq_of_track(tgt_seq, 'i-128')
        drum_wer = metric.calculate_wer(out_pitch, tgt_pitch)
        ret['drum_wer'] = drum_wer

        ''' Bar-level evaluation '''
        # Pad to 4 bars
        out_bar_cnt = out_seq.count('b-1')
        if out_bar_cnt < 4:
            out_seq += ['b-1'] * (4 - out_bar_cnt)

        out_bar1_idx = out_seq.index('b-1')
        out_bar2_idx = out_seq.index('b-1', out_bar1_idx+1)
        out_bar3_idx = out_seq.index('b-1', out_bar2_idx+1)
        out_bar4_idx = out_seq.index('b-1', out_bar3_idx+1)
        out_bar1 = out_seq[:out_bar1_idx+1]
        out_bar2 = out_seq[out_bar1_idx+1:out_bar2_idx+1]
        out_bar3 = out_seq[out_bar2_idx+1:out_bar3_idx+1]
        out_bar4 = out_seq[out_bar3_idx+1:out_bar4_idx+1]
        tgt_bar1_idx = tgt_seq.index('b-1')
        tgt_bar2_idx = tgt_seq.index('b-1', tgt_bar1_idx+1)
        tgt_bar3_idx = tgt_seq.index('b-1', tgt_bar2_idx+1)
        tgt_bar4_idx = tgt_seq.index('b-1', tgt_bar3_idx+1)
        tgt_bar1 = tgt_seq[:tgt_bar1_idx+1]
        tgt_bar2 = tgt_seq[tgt_bar1_idx+1:tgt_bar2_idx+1]
        tgt_bar3 = tgt_seq[tgt_bar2_idx+1:tgt_bar3_idx+1]
        tgt_bar4 = tgt_seq[tgt_bar3_idx+1:tgt_bar4_idx+1]
        for tgt_bar_seq, out_bar_seq in zip([tgt_bar1, tgt_bar2, tgt_bar3, tgt_bar4], [out_bar1, out_bar2, out_bar3, out_bar4]):
            # Note F1 (16th note quantized)
            note_f1 = metric.calculate_note_f1_q16(out_bar_seq, tgt_bar_seq)
            ret['note_f1'] = note_f1

        return ret


class LitM2mInstPred(L.LightningModule):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()

        # n_labels = 35 # Slakh quantized instruments
        n_labels = 129 # Raw midi program
        self.model = M2mClsModelSingleHead(pt_ckpt, n_class=n_labels, random_init=hparams['random_init'])

        if 'freeze_transformer' in hparams and hparams['freeze_transformer'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels)

        sch = self.lr_schedulers()
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # LR scheduler update
        if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels)

        ''' Calculate metrics '''
        metric = Metric()
        pred = torch.argmax(out, dim=1)

        # Compute the accuracy
        acc = (pred == labels).float().mean().item()
        metric.update('acc', acc)

        # Compute the top-3 accuracy
        top3 = torch.topk(out, 3, dim=1).indices
        correct = top3.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc = correct.mean().item()
        metric.update('top3_acc', top3_acc)

        # Compute the top-5 accuracy
        top5 = torch.topk(out, 5, dim=1).indices
        correct = top5.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-5 predictions
        top5_acc = correct.mean().item()
        metric.update('top5_acc', top5_acc)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = batch_tokenized.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def test_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels)

        ''' Calculate metrics '''
        metric = Metric()
        pred = torch.argmax(out, dim=1)

        # Compute the accuracy
        acc = (pred == labels).float().mean().item()
        metric.update('acc', acc)

        # Compute the top-3 accuracy
        top3 = torch.topk(out, 3, dim=1).indices
        correct = top3.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc = correct.mean().item()
        metric.update('top3_acc', top3_acc)

        # Compute the top-5 accuracy
        top5 = torch.topk(out, 5, dim=1).indices
        correct = top5.eq(labels.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-5 predictions
        top5_acc = correct.mean().item()
        metric.update('top5_acc', top5_acc)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = batch_tokenized.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss
    
    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_loss"])


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # # Different learning rate for different parts of the model
        # optimizer = optim.AdamW([
        #     {'params': self.model.cls_head.parameters(), 'lr': float(self.config['lr']), 'weight_decay':self.config['weight_decay']},
        #     {'params': self.model.decoder.parameters(), 'lr': float(self.config['lr_musecoco']), 'weight_decay':self.config['weight_decay']},
        # ])

        # Linear scheduler
        max_steps = self.num_training_steps()
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=max_steps,
        )
        ret = {"optimizer": optimizer, "lr_scheduler": scheduler},

        # # Annealing
        # scheduler = ReduceLROnPlateauPatch(
        #     optimizer,
        #     mode='min',
        #     factor=0.1,
        #     patience=self.config['lr_anneal_patience'],
        #     verbose=True
        # )

        # ret = {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "valid_loss",
        #     },
        # }

        
        # ret = {"optimizer": optimizer}
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)


class LitM2mInstPredPool(LitM2mInstPred):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__(pt_ckpt, tokenizer, hparams)

        # n_labels = 35 # Slakh quantized instruments
        n_labels = 129 # Raw midi program
        # self.model = M2mClsModelSingleHead(pt_ckpt, n_class=n_labels, random_init=hparams['random_init'])
        self.model = M2mClsModelSingleHeadPool(pt_ckpt, n_class=n_labels, random_init=hparams['random_init'])

        print('Random init: ', hparams['random_init'])

        if 'freeze_transformer' in hparams and hparams['freeze_transformer'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)


class LitM2mInstPredPaired(LitM2mInstPred):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__(pt_ckpt, tokenizer, hparams)

        # n_labels = 35 # Slakh quantized instruments
        # n_labels = 129 # Raw midi program
        n_labels = 2
        self.model = M2mClsModelSingleHead(pt_ckpt, n_class=n_labels, random_init=hparams['random_init'])
        

        if 'freeze_transformer' in hparams and hparams['freeze_transformer'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def validation_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda()
        
        # Tokenize the batch
        batch_tokenized = self.tk(
            note_seqs, 
            return_tensors="pt",
            padding=True,
        )['input_ids'].cuda()

        # Add BOS token
        bos_tokens = torch.full(size=(batch_tokenized.shape[0], 1), fill_value=self.tk.bos_token_id, dtype=torch.long, device=batch_tokenized.device)
        batch_tokenized = torch.cat([bos_tokens, batch_tokenized], dim=1)

        out = self.model(batch_tokenized)
        loss = nn.functional.cross_entropy(out, labels)

        ''' Calculate metrics '''
        metric = Metric()
        pred = torch.argmax(out, dim=1)

        # Compute the accuracy
        acc = (pred == labels).float().mean().item()
        metric.update('acc', acc)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = batch_tokenized.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

class LitM2mInstPredPairedPool(LitM2mInstPredPaired):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__(pt_ckpt, tokenizer, hparams)

        # n_labels = 35 # Slakh quantized instruments
        # n_labels = 129 # Raw midi program
        n_labels = 2
        self.model = M2mClsModelSingleHeadPool(pt_ckpt, n_class=n_labels, random_init=hparams['random_init'])
        

        if 'freeze_transformer' in hparams and hparams['freeze_transformer'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)





class LitM2mChordPred(L.LightningModule):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__()
        n_labels = 35 # 35 types of instruments
        # self.model = M2mClsModel4PosDoubleHead(pt_ckpt, random_init=hparams['random_init'])
        self.model = M2mClsModel8Head(pt_ckpt, random_init=hparams['random_init'])
        # self.model = M2mClsModel8HeadPool(pt_ckpt, random_init=hparams['random_init'])

        if 'freeze_transformer' in hparams and hparams['freeze_transformer'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        root_labels = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~12
        type_labels = torch.tensor([i[2] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~9
        
        # Tokenize the batch
        inp_seqs = self.tk(
            note_seqs, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_len'],
        )['input_ids'].cuda()

        root_out, type_out = self.model(inp_seqs) # [bs, n_pos, n_labels]
        # Put n_labels to the 2nd dimension
        root_out = root_out.permute(0, 2, 1)
        type_out = type_out.permute(0, 2, 1)

        loss_root = nn.functional.cross_entropy(root_out, root_labels)
        loss_type = nn.functional.cross_entropy(type_out, type_labels)
        loss = (loss_root + loss_type) / 2

        sch = self.lr_schedulers()
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log('train_lr', sch.get_lr()[0])
        
        # LR scheduler update
        if not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels_root = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~12
        labels_type = torch.tensor([i[2] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~9
        
        # Tokenize the batch
        inp_seqs = self.tk(
            note_seqs, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_len'],
        )['input_ids'].cuda()

        root_out, type_out = self.model(inp_seqs) # [bs, n_pos, n_labels]
        # Put n_labels to the 2nd dimension
        root_out = root_out.permute(0, 2, 1) # [bs, n_labels, n_pos]
        type_out = type_out.permute(0, 2, 1)

        loss_root = nn.functional.cross_entropy(root_out, labels_root)
        loss_type = nn.functional.cross_entropy(type_out, labels_type)
        loss = (loss_root + loss_type) / 2

        ''' Calculate metrics '''
        # Flatten the out and tgt
        root_out = root_out.permute(0, 2, 1).reshape(-1, root_out.shape[1])
        type_out = type_out.permute(0, 2, 1).reshape(-1, type_out.shape[1])
        labels_root = labels_root.reshape(-1)
        labels_type = labels_type.reshape(-1)
        
        metric = Metric()
        pred_root = torch.argmax(root_out, dim=1)
        pred_type = torch.argmax(type_out, dim=1)

        # Compute the accuracy
        acc_root = (pred_root == labels_root).float().mean().item()
        acc_type = (pred_type == labels_type).float().mean().item()
        metric.update('acc_root', acc_root)
        metric.update('acc_type', acc_type)

        # Compute top3 accuracy for both root and type
        top3_root = torch.topk(root_out, 3, dim=1).indices
        correct_root = top3_root.eq(labels_root.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc_root = correct_root.mean().item()
        metric.update('top3_acc_root', top3_acc_root)
        
        top3_type = torch.topk(type_out, 3, dim=1).indices
        correct_type = top3_type.eq(labels_type.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc_type = correct_type.mean().item()
        metric.update('top3_acc_type', top3_acc_type)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = inp_seqs.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def test_step(self, batch, batch_idx):
        # Get the note sequence and label
        note_seqs = [' '.join(i[0]) for i in batch]
        labels_root = torch.tensor([i[1] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~12
        labels_type = torch.tensor([i[2] for i in batch], dtype=torch.long).cuda() # [bs, n_pos=4], value: 0~9
        
        # Tokenize the batch
        inp_seqs = self.tk(
            note_seqs, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['max_len'],
        )['input_ids'].cuda()

        root_out, type_out = self.model(inp_seqs) # [bs, n_pos, n_labels]
        # Put n_labels to the 2nd dimension
        root_out = root_out.permute(0, 2, 1) # [bs, n_labels, n_pos]
        type_out = type_out.permute(0, 2, 1)

        loss_root = nn.functional.cross_entropy(root_out, labels_root)
        loss_type = nn.functional.cross_entropy(type_out, labels_type)
        loss = (loss_root + loss_type) / 2

        ''' Calculate metrics '''
        # Flatten the out and tgt
        root_out = root_out.permute(0, 2, 1).reshape(-1, root_out.shape[1])
        type_out = type_out.permute(0, 2, 1).reshape(-1, type_out.shape[1])
        labels_root = labels_root.reshape(-1)
        labels_type = labels_type.reshape(-1)
        
        metric = Metric()
        pred_root = torch.argmax(root_out, dim=1)
        pred_type = torch.argmax(type_out, dim=1)

        # Compute the accuracy
        acc_root = (pred_root == labels_root).float().mean().item()
        acc_type = (pred_type == labels_type).float().mean().item()
        metric.update('acc_root', acc_root)
        metric.update('acc_type', acc_type)

        # Compute top3 accuracy for both root and type
        top3_root = torch.topk(root_out, 3, dim=1).indices
        correct_root = top3_root.eq(labels_root.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc_root = correct_root.mean().item()
        metric.update('top3_acc_root', top3_acc_root)
        
        top3_type = torch.topk(type_out, 3, dim=1).indices
        correct_type = top3_type.eq(labels_type.unsqueeze(1)).any(dim=1).float()  # Check if labels are in top-3 predictions
        top3_acc_type = correct_type.mean().item()
        metric.update('top3_acc_type', top3_acc_type)

        scores = metric.average()

        # Logging to TensorBoard (if installed) by default
        bs = inp_seqs.shape[0]
        self.log("valid_loss", loss, batch_size=bs)

        for k, v in scores.items():
            self.log(f'valid_{k}', v, batch_size=bs)

        return loss

    def on_validation_epoch_end(self):
        scheduler = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["valid_loss"])


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # # Different learning rate for different parts of the model
        # optimizer = optim.AdamW([
        #     {'params': self.model.cls_head.parameters(), 'lr': float(self.config['lr']), 'weight_decay':self.config['weight_decay']},
        #     {'params': self.model.decoder.parameters(), 'lr': float(self.config['lr_musecoco']), 'weight_decay':self.config['weight_decay']},
        # ])

        # Linear scheduler
        max_steps = self.num_training_steps()
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=max_steps,
        )
        ret = {"optimizer": optimizer, "lr_scheduler": scheduler},

        # # CyclicalLR
        # step_per_epoch = self.get_step_per_epoch()
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=1e-7,
        #     max_lr=self.config['lr'],
        #     step_size_up=step_per_epoch,
        #     cycle_momentum=False,
        # )

        # # Annealing
        # scheduler = ReduceLROnPlateauPatch(
        #     optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=self.config['lr_anneal_patience'],
        #     verbose=True
        # )

        # ret = {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "valid_loss",
        #     },
        # }

        
        # ret = {"optimizer": optimizer}
        return ret
    
    def num_training_steps(self) -> int:
        """Get number of training steps"""
        if self.trainer.max_steps > -1:
            return self.trainer.max_steps

        self.trainer.fit_loop.setup_data()
        dataset_size = len(self.trainer.train_dataloader)
        num_steps = dataset_size * self.trainer.max_epochs

        return num_steps

    def get_step_per_epoch(self):
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        self.trainer.fit_loop.setup_data()
        return len(self.trainer.train_dataloader)


class LitM2mChordPredPool(LitM2mChordPred):
    '''
    Lightning module for instrument prediction
    '''

    def __init__(self, pt_ckpt, tokenizer, hparams, infer=False):
        super().__init__(pt_ckpt, tokenizer, hparams, infer)

        self.model = M2mClsModel8HeadPool(pt_ckpt, random_init=hparams['random_init'])

        if 'freeze_transformer' in hparams and hparams['freeze_transformer'] is True:
            # Set self.model.decoder require_grad to False
            for param in self.model.decoder.parameters():
                param.requires_grad = False

        if 'ft_last_layer' in hparams and hparams['ft_last_layer'] is True:
            for param in self.model.decoder.layers[-1].parameters():
                param.requires_grad = True

        if 'train_emb' in hparams and hparams['train_emb'] is True:
            for param in self.model.decoder.embed_tokens.parameters():
                param.requires_grad = True

        self.tk = tokenizer
        self.config = hparams
        self.save_hyperparameters(hparams)


class LitM2mSourceSep(LitM2mLM):
    def validation_step(self, batch, batch_idx):
        bs = len(batch)
        tokenized = self.tk(
            batch, 
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=self.config['max_len'],
        )
        inp_seqs = tokenized['input_ids'].cuda()
        attn_masks = tokenized['attention_mask'].cuda()

        ''' Target sequence loss '''
        # Craft label: find the [SEP] token's position in each sample, and mask the tokens before [SEP]
        sep_idx = find_token(inp_seqs, self.tk.sep_token_id) # a more robust way
        labels_unshifted = inp_seqs.clone()
        for i in range(inp_seqs.shape[0]):
            labels_unshifted[i, :sep_idx[i]+1] = -100
        # Mask the pad tokens
        for i in range(inp_seqs.shape[0]):
            for j in range(sep_idx[i]+1, inp_seqs.shape[1]):
                if inp_seqs[i, j] == self.tk.pad_token_id:
                    labels_unshifted[i, j] = -100
        loss = self.model(inp_seqs, labels=labels_unshifted).loss

        # Logging to TensorBoard (if installed) by default
        self.log("valid_loss", loss, batch_size=bs)

        # Generation in validation
        test_tot_steps = len(self.trainer.val_dataloaders)
        test_interval = test_tot_steps // self.config['val_gen_n_samples']

        if batch_idx % test_interval == 0:
            metric = Metric()

            generate_kwargs = {
                'max_length': self.config['max_len'],
                'use_cache': True, 
                'do_sample': True, # Strategy: greedy sampling
                'bad_words_ids': [[self.tk.pad_token_id]],
            }
            inp_sep_idx = sep_idx[0]
            inp_seq_gen = inp_seqs[0, :][:inp_sep_idx+1]
            attn_mask = attn_masks[0, :][:inp_sep_idx+1]
            inp_len = inp_seq_gen.shape[0]

            gen_out = self.model.generate(
                inp_seq_gen.unsqueeze(0),
                pad_token_id=self.tk.pad_token_id,
                attention_mask=attn_mask.unsqueeze(0),
                **generate_kwargs
            )[0][inp_len:]

            # Get detokenized output and input sequence
            out_str = self.tk.decode(gen_out, skip_special_tokens=True)
            out_seq = out_str.strip().split(' ')
            inp_str = self.tk.decode(inp_seq_gen, skip_special_tokens=False)
            inp_seq = inp_str.strip().split(' ')

            # Get the target sequence
            tgt_seq_ids = inp_seqs[0, :][inp_sep_idx+1:]
            tgt_seq = self.tk.decode(tgt_seq_ids, skip_special_tokens=True).strip().split(' ')

            # Get target insts
            inst_start_idx = inp_seq.index('[INST]') + 1
            # inst_end_idx = inp_seq.index('[SEP]') # for 4-bar inst and voice control model
            if '[MELODY]' not in inp_seq:
                inst_end_idx = inp_seq.index('[PITCH]') if '[PITCH]' in inp_seq else inp_seq.index('[SEP]')  # for 4-bar model
            else:
                inst_end_idx = inp_seq.index('[MELODY]')
            tgt_insts = inp_seq[inst_start_idx:inst_end_idx]

            # Get output insts
            out_insts = remi_utils.from_remi_get_inst_and_voice(out_seq)
            
            # Calculate inst iou
            inst_iou = metric.calculate_inst_iou_from_inst(out_insts, tgt_insts)
            self.log('valid_inst_iou', inst_iou, batch_size=bs)

            # Calculate voice wer
            voice_wer = metric.calculate_wer(out_insts, tgt_insts)
            self.log('valid_voice_wer', voice_wer, batch_size=bs)

            # Get bar count
            bar_count = out_seq.count('b-1')
            self.log('valid_bar_count', bar_count, batch_size=bs)

            # Groove similarity
            pos_wer, pos_sor = metric.calculate_groove_wer_sor(out_seq, tgt_seq)
            self.log('valid_pos_wer', pos_wer, batch_size=bs)

            # Pitch WER
            pitch_wer = metric.calculate_pitch_wer(out_seq, tgt_seq)
            self.log('valid_pitch_wer', pitch_wer, batch_size=bs)

            # Pitch WER per pos
            pitch_wer_per_pos = metric.calculate_pitch_wer_per_pos(out_seq, tgt_seq)
            self.log('valid_pitch_wer_per_pos', pitch_wer_per_pos, batch_size=bs)


        return loss

    def calculate_metrics(self, inp_seq, tgt_seq, out_seq) -> dict:
        '''
        Calculate the metrics for a single sample
        '''
        metric = Metric()
        ret = {}

        # Get target and output insts
        tgt_insts = remi_utils.from_remi_get_inst_and_voice(tgt_seq)
        out_insts = remi_utils.from_remi_get_inst_and_voice(out_seq)
        
        # Calculate inst iou
        inst_iou = metric.calculate_inst_iou_from_inst(out_insts, tgt_insts)
        ret['inst_iou'] = inst_iou

        # Voice WER
        voice_wer = metric.calculate_wer(out_insts, tgt_insts)
        ret['voice_wer'] = voice_wer

        # Pitch sequence similarity
        pitch_wer = metric.calculate_pitch_wer(out_seq, tgt_seq)
        ret['pitch_wer'] = pitch_wer

        pitch_iou = metric.calculate_pitch_iou(out_seq, tgt_seq)
        ret['pitch_iou'] = pitch_iou

        # Groove similarity
        pos_wer, pos_sor = metric.calculate_groove_wer_sor_mbar(out_seq, tgt_seq)
        ret['pos_wer'] = pos_wer

        pos_iou = metric.calculate_groove_iou_mbar(out_seq, tgt_seq)
        ret['pos_iou'] = pos_iou

        # Melody recall
        melody_recall = metric.calculate_melody_recall_mbar(out_seq, tgt_seq)
        ret['melody_recall'] = melody_recall

        ''' Track-wise metrics '''
        # Track-wise pitch sequence wer
        track_pitch_wer = metric.calculate_avg_track_pitch_wer(out_seq, tgt_seq)
        ret['track_pitch_wer'] = track_pitch_wer
        
        # Track-wise pitch sequence wer
        track_pitch_iou = metric.calculate_avg_track_pitch_iou(out_seq, tgt_seq)
        ret['track_pitch_iou'] = track_pitch_iou

        # # Track-wise groove similarity
        # track_pos_wer = metric.calculate_avg_track_pos_wer(out_seq, tgt_seq)
        # ret['track_pos_wer'] = track_pos_wer

        # Track-wise groove similarity
        track_pos_iou = metric.calculate_avg_track_pos_iou(out_seq, tgt_seq)
        ret['track_pos_iou'] = track_pos_iou

        # Duration difference
        dur_diff = metric.calculate_dur_dif_per_track(out_seq, tgt_seq)
        ret['dur_diff'] = dur_diff
        
        return ret
    

class LitM2mPianoReduction(LitM2mLM):
    def calculate_metrics(self, inp_seq, tgt_seq, out_seq) -> dict:
        '''
        Calculate the metrics for a single sample
        '''
        metric = Metric()
        ret = {}

        # Note F1
        note_f1 = metric.calculate_note_f1_q16(out_seq, tgt_seq)
        ret['note_f1'] = note_f1

        # Pitch sequence similarity
        pitch_wer = metric.calculate_pitch_wer(out_seq, tgt_seq)
        ret['pitch_wer'] = pitch_wer

        pitch_iou = metric.calculate_pitch_iou(out_seq, tgt_seq)
        ret['pitch_iou'] = pitch_iou

        # Groove similarity
        pos_wer, pos_sor = metric.calculate_groove_wer_sor_mbar(out_seq, tgt_seq)
        ret['pos_wer'] = pos_wer

        pos_iou = metric.calculate_groove_iou_mbar(out_seq, tgt_seq)
        ret['pos_iou'] = pos_iou

        return ret
    


class ReduceLROnPlateauPatch(ReduceLROnPlateau, _LRScheduler):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]
