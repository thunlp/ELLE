import collections
import sys
import torch.nn as nn
import torch
import math

def main():
    print(sys.argv[1])
    ckpt = torch.load(sys.argv[1])
    width_enlarge = True
    layer_enlarge = True
    enlarge_layer_num = 6
    enlarge_dim = 384
    attention_head = 12
    emb_num = ckpt['model']['decoder.sentence_encoder.embed_tokens.weight'].size()[0]
    width = ckpt['model']['decoder.sentence_encoder.embed_tokens.weight'].size()[1]
    inner_hidden = ckpt['model']['decoder.sentence_encoder.layers.0.fc1.weight'].size()[0]
    lst = []
    source_per_head = int(width / attention_head)
    target_per_head = int((width + enlarge_dim) / attention_head)

    if width_enlarge:
        for k, v in ckpt['model'].items():
            if 'embed_tokens' in k:
                new_weight = torch.empty((emb_num, width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                nn.init.normal_(new_weight, std=0.01)
                for i in range(attention_head):
                    new_weight[:, i * target_per_head: i * target_per_head + source_per_head] = v[:, i * source_per_head: (i+1) * source_per_head]
                lst.append([k, new_weight.clone()])
            elif 'embed_positions' in k:
                new_weight = torch.empty((514, width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                nn.init.normal_(new_weight, std=0.01)
                for i in range(attention_head):
                    new_weight[:, i * target_per_head: i * target_per_head + source_per_head] = v[:, i * source_per_head: (i+1) * source_per_head]
                lst.append([k, new_weight.clone()])
            elif 'emb_layer_norm' in k:
                new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                nn.init.normal_(new_weight, std=0.01)
                for i in range(attention_head):
                    new_weight[i * target_per_head: i * target_per_head + source_per_head] = v[i * source_per_head: (i+1) * source_per_head]
                lst.append([k, new_weight.clone()])
            elif 'encoder.layers' in k:
                k_split = k.split('.')
                if 'k_proj' in k_split[5] or 'q_proj' in k_split[5]:
                    if k_split[6] == 'weight':
                        new_weight = torch.empty((width + enlarge_dim, width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        for i in range(attention_head):
                            for j in range(attention_head):
                                new_weight[i * target_per_head: i * target_per_head + source_per_head, j * target_per_head: j * target_per_head + source_per_head] = v[i * source_per_head: i * source_per_head + source_per_head, j * source_per_head: (j+1) * source_per_head]
                        lst.append([k, new_weight.clone()])
                    elif k_split[6] == 'bias':
                        new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        for i in range(attention_head):
                            new_weight[i * target_per_head: i * target_per_head + source_per_head] = v[i * source_per_head: (i+1) * source_per_head]
                        lst.append([k, new_weight.clone()])
                    else:
                        print('1')
                        print(k)
                        print('wrong')
                elif 'v_proj' in k_split[5] or 'out_proj' in k_split[5]:
                    if k_split[6] == 'weight':
                        new_weight = torch.empty((width + enlarge_dim, width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        for i in range(attention_head):
                            for j in range(attention_head):
                                new_weight[i * target_per_head: i * target_per_head + source_per_head, j * target_per_head: j * target_per_head + source_per_head] = v[i * source_per_head: i * source_per_head + source_per_head, j * source_per_head: (j+1) * source_per_head]
                        lst.append([k, new_weight.clone()])
                    elif k_split[6] == 'bias':
                        new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        for i in range(attention_head):
                            new_weight[i * target_per_head: i * target_per_head + source_per_head] = v[i * source_per_head: (i+1) * source_per_head]
                        lst.append([k, new_weight.clone()])
                    else:
                        print('1')
                        print(k)
                        print('wrong')
                elif 'layer_norm' in k_split[4]:
                    new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(attention_head):
                        new_weight[i * target_per_head: i * target_per_head + source_per_head] = v[i * source_per_head: (i+1) * source_per_head]
                    lst.append([k, new_weight.clone()])
                if 'fc1' in k_split[4]:
                    if k_split[5] == 'weight':
                        new_weight = torch.empty((inner_hidden, width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        for i in range(attention_head):
                            new_weight[:, i * target_per_head: i * target_per_head + source_per_head] = v[:, i * source_per_head: (i+1) * source_per_head]
                        lst.append([k, new_weight.clone()])
                    elif k_split[5] == 'bias':
                        new_weight = torch.empty((inner_hidden), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        new_weight[: inner_hidden] = v
                        lst.append([k, new_weight.clone()])
                elif 'fc2' in k_split[4]:
                    if k_split[5] == 'weight':
                        new_weight = torch.empty((width + enlarge_dim, inner_hidden), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        for i in range(attention_head):
                            new_weight[i * target_per_head: i * target_per_head + source_per_head, :] = v[i * source_per_head: (i+1) * source_per_head, :]
                        lst.append([k, new_weight.clone()])
                    elif k_split[5] == 'bias':
                        new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        for i in range(attention_head):
                            new_weight[i * target_per_head: i * target_per_head + source_per_head] = v[i * source_per_head: (i+1) * source_per_head]
                        lst.append([k, new_weight.clone()])
                    else:
                        print('2')
                        print(k)
                        print('wrong')
            elif 'lm_head' in k:
                if 'dense' in k:
                    if 'weight' in k:
                        new_weight = torch.empty((width + enlarge_dim, width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        lst.append([k, new_weight.clone()])
                    elif 'bias' in k:
                        new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        lst.append([k, new_weight.clone()])
                    else:
                        print('3')
                        print(k)
                        print('wrong')
                elif 'layer_norm' in k:
                    if 'weight' in k:
                        new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        lst.append([k, new_weight.clone()])
                    elif 'bias' in k:
                        new_weight = torch.empty((width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        lst.append([k, new_weight.clone()])
                    else:
                        print('3')
                        print(k)
                        print('wrong')
                else:
                    if 'weight' in k:
                        new_weight = torch.empty((emb_num, width + enlarge_dim), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        lst.append([k, new_weight.clone()])
                    elif 'bias' in k:
                        new_weight = torch.empty((emb_num), dtype = torch.float16, device = 'cuda')
                        nn.init.normal_(new_weight, std=0.01)
                        lst.append([k, new_weight.clone()])
                    else:
                        print('4')
                        print(k)
                        print('wrong')
            else:
                print('5')
                print(k)
                print('wrong')

    for k, v in lst:
        ckpt['model'][k] = v

    if layer_enlarge:
        lst = []
        for k,v in ckpt['model'].items():
            if "encoder.layers" in k:
                k_split = k.split('.')
                l_id = int(k_split[3])
                if l_id > ckpt['args'].encoder_layers - enlarge_layer_num:
                    k_split[3] = str(l_id + enlarge_layer_num)
                    new_k = '.'.join(k_split)
                    lst.append([new_k, v.clone()])

        for k, v in lst:
            ckpt['model'][k] = v

        ckpt['args'].encoder_layers += enlarge_layer_num

    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()
