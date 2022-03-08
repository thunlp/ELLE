"""
preprocessing script before training distillBert
specific to bert->distillbert
"""
import argparse
import os
import math
from typing import NewType, NoReturn
import torch
import numpy as np

from transformer.modeling import BertForPreTraining

def wider3d(w,dim,new_width,choices,div=False):
	old_width = w.size(dim)
	if dim==0:
		new_w = torch.randn(new_width,w.size(1),w.size(2))
	elif dim ==1:
		new_w = torch.randn(w.size(0),new_width,w.size(2))
	else:
		new_w = torch.randn(w.size(0),w.size(1),new_width)
	new_w.narrow(dim,0,old_width).copy_(w)
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

	return new_w

def wider2d(w,dim,new_width,choices,div=False):
	old_width = w.size(dim)
	if dim==0:
		new_w = torch.randn(new_width,w.size(1))
	else:
		new_w = torch.randn(w.size(0),new_width)
	new_w.narrow(dim,0,old_width).copy_(w)
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
	return new_w

def wider(w,new_width,choices,div=False):
	old_width = w.size(0)
	new_w = torch.randn(new_width)
	new_w.narrow(0,0,old_width).copy_(w)
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
	return new_w

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

a = torch.randn(3,2,2)
choices = get_choices(2,4)
b = wider3d(w=a, dim=1, new_width=4, choices=choices, div=True)

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='bert2bertFKI')
	parser.add_argument('--bm_path', type=str)
	parser.add_argument('--sm_path', type=str)
	parser.add_argument('--is_always_left', action='store_true')
	parser.add_argument('--bert2deeperbert_random_pad', action='store_true')
	args = parser.parse_args()	

	prefix = 'bert'
	bm = BertForPreTraining.from_scratch(args.bm_path)
	######sm = BertForPreTraining.from_scratch(args.sm_path)
	sm = BertForPreTraining.from_pretrained(args.sm_path)

	bm_layers = bm.config.num_hidden_layers
	bm_hidden = bm.config.hidden_size
	bm_num_heads = bm.config.num_attention_heads
	bm_intermediate_size = bm.confid.intermediate_size
	sm_layers = sm.config.num_hidden_layers
	sm_hidden = sm.config.hidden_size
	sm_num_heads = sm.config.num_attention_heads
	sm_intermediate_size = sm.confid.intermediate_size
	headdim = bm_hidden // bm_num_heads
	vocab_size = bm.config.vocab_size

	sm_dict = sm.state_dict()
	bm_dict = bm.state_dict()

	x = bm_hidden / sm_hidden

	if not args.bert2deeperbert_random_pad:
		sm_layer_idxs = [i for i in range(sm_layers)]
		sm_layer_idx_for_bert2bert_top = []
		n_times = bm_layers // sm_layers
		sm_layer_idx_for_bert2bert_top.extend(sm_layer_idxs * n_times)
		top_layers = bm_layers % sm_layers
		if top_layers !=0:
			sm_layer_idx_for_bert2bert_top.extend(sm_layer_idxs[-top_layers:])
		print(sm_layer_idx_for_bert2bert_top)
		print('default:bert2deepbert_top')

	choose_hidden_dims = get_choices(sm_hidden,bm_hidden,is_always_left=args.is_always_left)

	######for w in ['word_embeddings','position_embeddings','token_embeddings']:
	for w in ['word_embeddings','position_embeddings','token_type_embeddings']:
		bm_dict[f'bert.embeddings.{w}.weight'].copy_(wider2d(sm_dict[f'bert.embeddings.{w}.weight'],1,bm_hidden,choose_hidden_dims))
	for w in ['weight','bias']:
		######bm_dict[f'bert.embeddings.LayerNorm.{w}'].copy_(wider2d(sm_dict[f'bert.embeddings.LayerNorm.{w}'],bm_hidden,choose_hidden_dims))
		bm_dict[f'bert.embeddings.LayerNorm.{w}'].copy_(wider(sm_dict[f'bert.embeddings.LayerNorm.{w}'],bm_hidden,choose_hidden_dims))

	sm_layer_idx = 0
	for bm_layer_idx in range(bm_layers):
		if args.bert2deeperbert_random_pad:
			if bm_layer_idx < sm_layers:
				sm_layer_idx = bm_layer_idx
			else:
				break
		else:
			sm_layer_idx = sm_layer_idx_for_bert2bert_top[bm_layer_idx]

		#small sm_layer_idx -> big bm_layeridx

	bel = 'bert.encoder.layer'

	choose_heads = get_choices(sm_num_heads,bm_num_heads,is_always_left=args.is_always_left)
	######choose_mlp_dim = get_choices(sm_intermediate_size,bm_intermediate_size,is_always_left=args.is_always_left)
	choose_mlp_dims = get_choices(sm_intermediate_size,bm_intermediate_size,is_always_left=args.is_always_left)

	bm_dict[f'{bel}.{bm_layer_idx}.attention.self.query.weight'].reshape(bm_num_heads,headdim,bm_hidden)\
		.permute(0,2,1)\
		.copy_(wider3d(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.self.query.weight'],1,bm_hidden,
				choices=choose_hidden_dims,div=True)
			.reshape(sm_num_heads,headdim,bm_hidden)
			.permute(0,2,1),0,bm_num_heads,choices=choose_heads))

	bm_dict[f'{bel}.{bm_layer_idx}.attention.self.query.bias']\
		.reshape(bm_num_heads,headdim)\
		.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.self.query.bias'].reshape(sm_num_heads,headdim),
			dim=0,new_width=bm_num_heads,choices=choose_heads))

	bm_dict[f'{bel}.{bm_layer_idx}.attention.self.key.weight'].reshape(bm_num_heads,headdim,bm_hidden)\
		.permute(0,2,1)\
		.copy_(wider3d(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.self.key.weight'],1,bm_hidden,
				choices=choose_hidden_dims,div=True)
			.reshape(sm_num_heads,headdim,bm_hidden)
			.permute(0,2,1),0,bm_num_heads,choices=choose_heads))

	bm_dict[f'{bel}.{bm_layer_idx}.attention.self.key.bias']\
		.reshape(bm_num_heads,headdim)\
		.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.self.key.bias'].reshape(sm_num_heads,headdim),
			dim=0,new_width=bm_num_heads,choices=choose_heads))

	bm_dict[f'{bel}.{bm_layer_idx}.attention.self.value.weight'].reshape(bm_num_heads,headdim,bm_hidden)\
		.permute(0,2,1)\
		.copy_(wider3d(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.self.value.weight'],1,bm_hidden,
				choices=choose_hidden_dims,div=True)
			.reshape(sm_num_heads,headdim,bm_hidden)
			.permute(0,2,1),0,bm_num_heads,choices=choose_heads))

	bm_dict[f'{bel}.{bm_layer_idx}.attention.self.value.bias']\
		.reshape(bm_num_heads,headdim)\
		.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.self.value.bias'].reshape(sm_num_heads,headdim),
			dim=0,new_width=bm_num_heads,choices=choose_heads))

	bm_dict[f'{bel}.{bm_layer_idx}.attention.output.dense.weight'].reshape(bm_hidden,bm_num_heads,headdim)\
		.permute(1,2,0)\
		.copy_(wider3d(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.output.dense.weight'],dim=0,
				new_width=bm_hidden,choices=choose_hidden_dims)
			.reshape(bm_hidden,sm_num_heads,headdim)
			.permute(1,2,0),dim=0,new_width=bm_num_heads,choices=choose_heads,div=True))

		######.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.output.dense.bias'],
	bm_dict[f'{bel}.{bm_layer_idx}.attention.output.dense.bias']\
		.copy_(wider(sm_dict[f'{bel}.{sm_layer_idx}.attention.output.dense.bias'],
			new_width=bm_hidden,choices=choose_hidden_dims))

		######.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.output.LayerNorm.weight'],
	bm_dict[f'{bel}.{bm_layer_idx}.attention.output.LayerNorm.weight']\
		.copy_(wider(sm_dict[f'{bel}.{sm_layer_idx}.attention.output.LayerNorm.weight'],
			new_width=bm_hidden,choices=choose_hidden_dims))	
		
		######.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.attention.output.LayerNorm.bias'],
	bm_dict[f'{bel}.{bm_layer_idx}.attention.output.LayerNorm.bias']\
		.copy_(wider(sm_dict[f'{bel}.{sm_layer_idx}.attention.output.LayerNorm.bias'],
			new_width=bm_hidden,
			choices=choose_hidden_dims))
	
	bm_dict[f'{bel}.{bm_layer_idx}.intermediate.dense.weight']\
		.copy_(wider2d(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.intermediate.dense.weight'],
			dim=1,new_width=bm_hidden,choices=choose_hidden_dims,div=True),
			dim=0,new_width=bm_intermediate_size,choice=choose_mlp_dims))

	bm_dict[f'{bel}.{bm_layer_idx}.intermediate.dense.bias']\
		.copy_(wider(sm_dict[f'{bel}.{sm_layer_idx}.intermediate.dense.bias'],
		new_width=bm_intermediate_size,choices=choose_mlp_dims))

	bm_dict[f'{bel}.{bm_layer_idx}.output.dense.weight']\
		.copy_(wider2d(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.output.dense.weight'],
			dim=0,new_width=bm_hidden,choices=choose_hidden_dims),
		dim=1,new_width=bm_intermediate_size,choice=choose_mlp_dims,div=True))

	bm_dict[f'{bel}.{bm_layer_idx}.output.dense.bias']\
		.copy_(wider(sm_dict[f'{bel}.{sm_layer_idx}.output.dense.bias'],
		new_width=bm_hidden,choices=choose_hidden_dims))

		######.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.output.LayerNorm.weight'],
	bm_dict[f'{bel}.{bm_layer_idx}.output.LayerNorm.weight']\
		.copy_(wider(sm_dict[f'{bel}.{sm_layer_idx}.output.LayerNorm.weight'],
			new_width=bm_hidden,choices=choose_hidden_dims))	
		
		######.copy_(wider2d(sm_dict[f'{bel}.{sm_layer_idx}.output.LayerNorm.bias'],
	bm_dict[f'{bel}.{bm_layer_idx}.output.LayerNorm.bias']\
		.copy_(wider(sm_dict[f'{bel}.{sm_layer_idx}.output.LayerNorm.bias'],
			new_width=bm_hidden,choices=choose_hidden_dims))

	bpd = 'bert.pooler.dense'

	bm_dict[f'{bpd}.weight']\
		.copy_(wider2d(wider2d(sm_dict[f'{bpd}.weight'],dim=1,new_width=bm_hidden,choices=choose_hidden_dims,div=True),
			dim=0,new_width=bm_hidden,choices=choose_hidden_dims))

	bm_dict[f'{bpd}.bias']\
		.copy_(wider(sm_dict[f'{bpd}.bias'],new_width=bm_hidden,choices=choose_hidden_dims))

	bm_dict['cls.predictions.bias'].copy_(sm_dict['cls.predictions.bias'])

	cpt = 'cls.predictions.transform'

	bm_dict[f'{cpt}.dense.weight']\
		.copy_(wider2d(wider2d(sm_dict[f'{cpt}.dense.weight'],dim=1,new_width=bm_hidden,
			choices=choose_hidden_dims,div=True),
		dim=0,new_width=bm_hidden,choices=choose_hidden_dims))

	bm_dict[f'{cpt}.dense.bias']\
		.copy_(wider(sm_dict[f'{cpt}.dense.bias'],new_width=bm_hidden,choices=choose_hidden_dims))
	
		######.copy_(wider(sm_dict[f'{cpt}.LayerNorm'],new_width=bm_hidden,choices=choose_hidden_dims))
	bm_dict[f'{cpt}.LayerNorm.weight']\
		.copy_(wider(sm_dict[f'{cpt}.LayerNorm.weight'],new_width=bm_hidden,choices=choose_hidden_dims,div=True))

		######.copy_(wider(sm_dict[f'{bpd}.LayerNorm.bias'],new_width=bm_hidden,choices=choose_hidden_dims))
	bm_dict[f'{cpt}.LayerNorm.bias']\
		.copy_(wider(sm_dict[f'{cpt}.LayerNorm.bias'],new_width=bm_hidden,choices=choose_hidden_dims,div=True))

	bm_dict[f'cls.seq_relationship.weight']\
		.copy_(wider2d(sm_dict[f'cls.seq_relationship.weight'],dim=1,new_width=bm_hidden,
			choices=choose_hidden_dims,div=True))

	bm_dict[f'cls.seq_relationship.bias']\
		.copy_(sm_dict[f'cls.seq.relationship.bias'])

	torch.save(bm_dict,os.path.join(args.bm_path,'pytorch_model.bin'))