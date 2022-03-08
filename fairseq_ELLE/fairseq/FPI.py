"""
preprocessing script before training distillBert
specific to bert->distillbert
"""
import collections
import sys
import torch.nn as nn
import torch
import math
import numpy as np
import random
from . import checkpoint_utils

def wider3d(w,dim,new_width,choices,div=False):
	old_width = w.size(dim)
	if dim==0:
		new_w = torch.randn(new_width,w.size(1),w.size(2), dtype=torch.float16, device='cuda')
	elif dim ==1:
		new_w = torch.randn(w.size(0),new_width,w.size(2), dtype=torch.float16, device='cuda')
	else:
		new_w = torch.randn(w.size(0),w.size(1),new_width, dtype=torch.float16, device='cuda')
	new_w.narrow(dim,0,old_width).copy_(w.clone())
	tracking = dict()
	for i in range(old_width,new_width):
		idx = choices[i-old_width]
		try:
			tracking[idx].append(i)
		except:
			tracking[idx] = [idx]
			tracking[idx].append(i)

		new_w.select(dim,i).copy_(w.select(dim,idx).clone())
	if div:
		if dim == 0:
			for idx,d in tracking.items():
				for item in d:
					new_w[item].div_(len(d))
		elif dim == 1:
			for idx,d in tracking.items():
				for item in d:
					new_w[:,item].div_(len(d))
		else:
			for idx,d in tracking.items():
				for item in d:
					######new_w[:,:,item].div(len(d))
					new_w[:,:,item].div_(len(d))

	return new_w.half()

def wider2d(w,dim,new_width,choices,div=False):
	old_width = w.size(dim)
	if dim==0:
		new_w = torch.randn(new_width, w.size(1), dtype=torch.float16, device='cuda')
	else:
		new_w = torch.randn(w.size(0), new_width, dtype=torch.float16, device='cuda')
	new_w.narrow(dim,0,old_width).copy_(w.clone())
	tracking = dict()
	for i in range(old_width,new_width):
		idx = choices[i-old_width]
		try:
			tracking[idx].append(i)
		except:
			tracking[idx] = [idx]
			tracking[idx].append(i)

		new_w.select(dim,i).copy_(w.select(dim,idx).clone())
	if div:
		if dim == 0:
			for idx,d in tracking.items():
				for item in d:
					new_w[item].div_(len(d))
		else:
			for idx,d in tracking.items():
				for item in d:
					new_w[:,item].div_(len(d))
	return new_w.half()

def wider(w,new_width,choices,div=False):
	old_width = w.size(0)
	new_w = torch.randn(new_width, dtype=torch.float16, device='cuda')
	new_w.narrow(0,0,old_width).copy_(w.clone())
	tracking = dict()
	for i in range(old_width,new_width):
		idx = choices[i-old_width]
		try:
			tracking[idx].append(i)
		except:
			tracking[idx] = [idx]
			tracking[idx].append(i)

		new_w.select(0,i).copy_(w.select(0,idx).clone())
	if div:
		for idx,d in tracking.items():
			for item in d:
				new_w[item].div_(len(d))
	return new_w.half()

def get_choices(old_width,new_width,is_always_left=False):
	choices = []
	if is_always_left:
		idx = 0
		for i in range(old_width, new_width):
			choices.append(idx)
			idx += 1
	else:
		for i in range(old_width,new_width):
			idx = np.random.randint(0,old_width)
			choices.append(idx)
	return choices

def FPI(path, bm_layers, bm_hidden, bm_num_heads, bm_intermediate_size, save_path, add_noise=False, add_last=False):
	#ckpt = checkpoint_utils.load_checkpoint_to_cpu(path)
	ckpt = torch.load(path)
	random.seed(ckpt['args'].seed)
	np.random.seed(ckpt['args'].seed)
	sm_layers = ckpt['args'].encoder_layers
	sm_hidden = ckpt['args'].encoder_embed_dim
	sm_num_heads = ckpt['args'].encoder_attention_heads
	sm_intermediate_size = ckpt['args'].encoder_ffn_embed_dim
	is_always_left = ckpt['args'].is_always_left
	layer_candidates = ckpt['args'].layer_candidates
	layer_idxs = ckpt['args'].layer_idxs
	headdim = bm_hidden // bm_num_heads
	vocab_size = ckpt['model']['decoder.sentence_encoder.embed_tokens.weight'].size()[0]

	if layer_candidates == None:
		layer_candidates = [i for i in range(sm_layers)]
	if layer_idxs == None:
		layer_idxs = [i for i in range(sm_layers)]
	new_layer_idxs = []
	
	added_layers = []
	added_layer_num = bm_layers - sm_layers
	print("number of added layer: " + str(added_layer_num))
	print("old layer candidates: " + str(layer_candidates))
	print("old layer idxs: " + str(layer_idxs))
	if len(layer_candidates) >= added_layer_num:
		added_layers = layer_candidates[-added_layer_num:] if add_last else random.sample(layer_candidates, added_layer_num)
		for layer in added_layers: 
			layer_candidates.remove(layer)
	else:
		while add_layer_num > len(layer_candidates):
			added_layers = added_layer + layer_candidates.copy()
			add_layer_num -= len(layer_candidates)
			layer_candidates = [i for i in range(max(layer_idxs)+1)]
		new_added_layers = layer_candidates[-added_layer_num:] if add_last else random.sample(layer_candidates, added_layer_num)
		for layer in new_added_layers: 
			layer_candidates.remove(layer)
		added_layers = added_layers + new_added_layers
			
	sm_layer_idxs = [i for i in range(sm_layers)]
	sm_layer_idx_for_bert2bert_top = []
	new_layer_idxs = []
	for layer in sm_layer_idxs:
		idx = layer_idxs[layer]
		sm_layer_idx_for_bert2bert_top.append(layer)
		new_layer_idxs.append(idx)
		while idx in added_layers:
			sm_layer_idx_for_bert2bert_top.append(layer)
			new_layer_idxs.append(idx)
			added_layers.remove(idx)
	assert len(new_layer_idxs) == bm_layers
	assert len(new_layer_idxs) == len(sm_layer_idx_for_bert2bert_top)
	print("new layer candidates: " + str(layer_candidates))
	print("new layer idxs: " + str(new_layer_idxs))
	if len(new_layer_idxs) % (max(layer_idxs)+1) == 0:
		new_layer_idxs = [i for i in range(bm_layers)]
		layer_candidates = [i for i in range(bm_layers)]
	
	print("fianl layer candidates: " + str(layer_candidates))
	print("final layer idxs: " + str(new_layer_idxs))
	'''
	sm_layer_idxs = [i for i in range(sm_layers)]
	sm_layer_idx_for_bert2bert_top = []
	n_times = bm_layers // sm_layers
	sm_layer_idx_for_bert2bert_top.extend(sm_layer_idxs * n_times)
	top_layers = bm_layers % sm_layers
	if top_layers !=0:
		sm_layer_idx_for_bert2bert_top.extend(sm_layer_idxs[-top_layers:])
	print(sm_layer_idx_for_bert2bert_top)
	print('default:bert2deepbert_top')
	'''
	print("sm layer idx for bert2bert: " + str(sm_layer_idx_for_bert2bert_top))


	choose_hidden_dims = get_choices(sm_hidden, bm_hidden, is_always_left=is_always_left)

	lst = []
	for k, v in ckpt['model'].items():
		if 'embed_tokens' in k or 'embed_position' in k or 'domain_embeddings' in k:
			new_weight = wider2d(v, 1, bm_hidden, choose_hidden_dims)
		elif 'emb_layer_norm' in k:
			new_weight = wider(v, bm_hidden, choose_hidden_dims)
		elif 'lm_head' in k:
			if 'dense' in k:
				if 'weight' in k:
					new_weight = wider2d(wider2d(v, 1, bm_hidden, choose_hidden_dims, div=True), 0, bm_hidden, choices=choose_hidden_dims)
				elif 'bias' in k:
					new_weight = wider(v, bm_hidden, choose_hidden_dims)
			elif 'layer_norm' in k:
				new_weight = wider(v, bm_hidden, choose_hidden_dims, div=True)
			elif 'weight' in k:
				new_weight = wider2d(v, 1, bm_hidden, choose_hidden_dims, div=True)
			elif 'bias' in k:
				new_weight = v
		#print(k)		
		lst.append([k, new_weight.clone()])

	sm_layer_idx = 0
	for bm_layer_idx in range(bm_layers):
		sm_layer_idx = sm_layer_idx_for_bert2bert_top[bm_layer_idx]
	
		choose_heads = get_choices(sm_num_heads, bm_num_heads, is_always_left=is_always_left)
		choose_mlp_dims = get_choices(sm_intermediate_size,bm_intermediate_size,is_always_left=is_always_left)

		layer = f'decoder.sentence_encoder.layers.{sm_layer_idx}'
		new_layer = f'decoder.sentence_encoder.layers.{bm_layer_idx}'

		#self attention
		for w in ['q_proj', 'k_proj', 'v_proj']:
			
			k = f'{layer}.self_attn.{w}.weight'
			v = ckpt['model'][k]

			new_weight = torch.zeros((bm_hidden, bm_hidden), dtype=torch.float16, device='cuda')
			new_weight.reshape(bm_num_heads, headdim, bm_hidden).permute(0,2,1).copy_(wider3d(wider2d(v, 1, bm_hidden, choices=choose_hidden_dims, div=True).reshape(sm_num_heads, headdim, bm_hidden).permute(0,2,1), 0, bm_num_heads, choices=choose_heads))

			new_k = f'{new_layer}.self_attn.{w}.weight'
			#print(new_k)
			lst.append([new_k, new_weight.clone()])

			k = f'{layer}.self_attn.{w}.bias'
			v = ckpt['model'][k]

			new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
			new_weight.reshape(bm_num_heads, headdim).copy_(wider2d(v.reshape(sm_num_heads, headdim), 0, bm_num_heads, choose_heads))
			
			new_k = f'{new_layer}.self_attn.{w}.bias'
			#print(new_k)
			lst.append([new_k, new_weight.clone()])


		k = f'{layer}.self_attn.out_proj.weight'
		v = ckpt['model'][k]
		new_weight = torch.zeros((bm_hidden, bm_hidden), dtype=torch.float16, device='cuda')
		new_weight.reshape(bm_hidden, bm_num_heads, headdim).permute(1,2,0).copy_(wider3d(wider2d(v, 0, bm_hidden, choose_hidden_dims).reshape(bm_hidden, sm_num_heads, headdim).permute(1,2,0), 0, bm_num_heads, choose_heads, div=True))

		new_k = f'{new_layer}.self_attn.out_proj.weight'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		k = f'{layer}.self_attn.out_proj.bias'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims))

		new_k = f'{new_layer}.self_attn.out_proj.bias'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		k = f'{layer}.self_attn_layer_norm.weight'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims))

		new_k = f'{new_layer}.self_attn_layer_norm.weight'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		k = f'{layer}.self_attn_layer_norm.bias'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims))

		new_k = f'{new_layer}.self_attn_layer_norm.bias'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		#ffn
		k = f'{layer}.fc1.weight'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_intermediate_size, bm_hidden, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider2d(wider2d(v, 1, bm_hidden, choose_hidden_dims, div=True), 0, bm_intermediate_size, choose_mlp_dims))

		new_k = f'{new_layer}.fc1.weight'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		k = f'{layer}.fc1.bias'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_intermediate_size, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider(v, bm_intermediate_size, choose_mlp_dims))

		new_k = f'{new_layer}.fc1.bias'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])


		k = f'{layer}.fc2.weight'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_hidden, bm_intermediate_size, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider2d(wider2d(v, 0, bm_hidden, choose_hidden_dims), 1, bm_intermediate_size, choose_mlp_dims, div=True))

		new_k = f'{new_layer}.fc2.weight'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		k = f'{layer}.fc2.bias'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims))

		new_k = f'{new_layer}.fc2.bias'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		#layer_norm
		k = f'{layer}.final_layer_norm.weight'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims))

		new_k = f'{new_layer}.final_layer_norm.weight'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])

		k = f'{layer}.final_layer_norm.bias'
		v = ckpt['model'][k]
		new_weight = torch.zeros(bm_hidden, dtype=torch.float16, device='cuda')
		new_weight.copy_(wider(v, bm_hidden, choose_hidden_dims))

		new_k = f'{new_layer}.final_layer_norm.bias'
		#print(new_k)
		lst.append([new_k, new_weight.clone()])
	
	for k, v in lst:
		noise = torch.zeros_like(v, dtype=torch.float16, device='cuda')
		if add_noise:
			nn.init.normal_(noise, std=0.01)
		ckpt['model'][k] = v + noise
	'''
	for k, v in ckpt['model'].items():
		print(k)
		print(v.size())
	'''
	
	ckpt['args'].encoder_layers = bm_layers
	ckpt['args'].encoder_embed_dim = bm_hidden
	ckpt['args'].encoder_attention_heads = bm_num_heads
	ckpt['args'].encoder_ffn_embed_dim = bm_intermediate_size
	ckpt['args'].arch = f'roberta_{bm_layers}layer_{bm_hidden}hidden_{bm_num_heads}head_{bm_intermediate_size}ffn'
	ckpt['args'].layer_candidates = layer_candidates
	ckpt['args'].layer_idxs = new_layer_idxs

		
	torch.save(ckpt, save_path)
	
	return layer_candidates, new_layer_idxs



		
			


			



	


	
	