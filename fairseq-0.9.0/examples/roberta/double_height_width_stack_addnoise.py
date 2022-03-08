import collections
import sys
import torch.nn as nn
import torch
import math

def main():
    ckpt = torch.load(sys.argv[1])

    whether_stack = False
    enlarge_n_times = 2
    emb_num = ckpt['model']['decoder.sentence_encoder.embed_tokens.weight'].size()[0]
    height = 6
    width = ckpt['model']['decoder.sentence_encoder.embed_tokens.weight'].size()[1]
    inner_hidden = ckpt['model']['decoder.sentence_encoder.layers.0.fc1.weight'].size()[0]
    lst = []
    
    eff1 = torch.sqrt(torch.tensor(enlarge_n_times, dtype = torch.float16, device='cuda')) / enlarge_n_times
    eff2 = torch.sqrt(eff1 * enlarge_n_times) / enlarge_n_times
    eff3 = torch.sqrt(eff1 * enlarge_n_times) / torch.sqrt(torch.tensor(enlarge_n_times, dtype = torch.float16, device='cuda'))

    for k, v in ckpt['model'].items():
        print(k)
        if 'embed_tokens' in k:
            new_weight = torch.empty((emb_num, width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
            nn.init.normal_(new_weight, std=0.01)
            for i in range(width * enlarge_n_times):
                new_weight[:, i] += v[:, math.floor(float(i) / enlarge_n_times)] * eff1
            lst.append([k, new_weight.clone()])
        elif 'embed_positions' in k:
            new_weight = torch.empty((514, width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
            nn.init.normal_(new_weight, std=0.01)
            for i in range(width * enlarge_n_times):
                new_weight[:, i] += v[:, math.floor(float(i) / enlarge_n_times)] * eff1
            lst.append([k, new_weight.clone()])
        elif 'emb_layer_norm' in k:
            new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
            nn.init.normal_(new_weight, std=0.01)
            for i in range(width * enlarge_n_times):
                new_weight[i] += v[math.floor(float(i) / enlarge_n_times)] * eff1
            lst.append([k, new_weight.clone()])
        elif 'encoder.layers' in k:
            k_split = k.split('.')
            if 'k_proj' in k_split[5] or 'q_proj' in k_split[5]:
                if k_split[6] == 'weight':
                    new_weight = torch.empty((width * enlarge_n_times, width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        for j in range(width * enlarge_n_times):
                            new_weight[i, j] += v[math.floor(float(i) / enlarge_n_times), math.floor(float(j) / enlarge_n_times)] * eff2
                    lst.append([k, new_weight.clone()])
                elif k_split[6] == 'bias':
                    new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[i] += v[math.floor(float(i) / enlarge_n_times)] * eff3
                    lst.append([k, new_weight.clone()])
                else:
                    print('1')
                    print(k)
                    print('wrong')
            elif 'v_proj' in k_split[5] or 'out_proj' in k_split[5]:
                if k_split[6] == 'weight':
                    new_weight = torch.empty((width * enlarge_n_times, width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        for j in range(width * enlarge_n_times):
                            new_weight[i, j] += v[math.floor(float(i) / enlarge_n_times), math.floor(float(j) / enlarge_n_times)] / enlarge_n_times
                    lst.append([k, new_weight.clone()])
                elif k_split[6] == 'bias':
                    new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[i] += v[math.floor(float(i) / enlarge_n_times)] * eff1
                    lst.append([k, new_weight.clone()])
                else:
                    print('1')
                    print(k)
                    print('wrong')
            elif 'layer_norm' in k_split[4]:
                new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                nn.init.normal_(new_weight, std=0.01)
                for i in range(width * enlarge_n_times):
                    new_weight[i] += v[math.floor(float(i) / enlarge_n_times)] * eff1
                lst.append([k, new_weight.clone()])
            if 'fc1' in k_split[4]:
                if k_split[5] == 'weight':
                    new_weight = torch.empty((inner_hidden, width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[:, i] += v[:, math.floor(float(i) / enlarge_n_times)] * eff1
                    lst.append([k, new_weight.clone()])
                elif k_split[5] == 'bias':
                    new_weight = torch.empty((inner_hidden), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    new_weight[: inner_hidden] += v
                    lst.append([k, new_weight.clone()])
            elif 'fc2' in k_split[4]:
                if k_split[5] == 'weight':
                    new_weight = torch.empty((width * enlarge_n_times, inner_hidden), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[i, : ] += v[math.floor(float(i) / enlarge_n_times), :] * eff1
                    lst.append([k, new_weight.clone()])
                elif k_split[5] == 'bias':
                    new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[i] += v[math.floor(float(i) / enlarge_n_times)] * eff1
                    lst.append([k, new_weight.clone()])
                else:
                    print('2')
                    print(k)
                    print('wrong')
        elif 'lm_head' in k:
            if 'dense' in k:
                if 'weight' in k:
                    new_weight = torch.empty((width * enlarge_n_times, width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        for j in range(width * enlarge_n_times):
                            new_weight[i, j] += v[math.floor(float(i) / enlarge_n_times), math.floor(float(j) / enlarge_n_times)] * eff1
                    lst.append([k, new_weight.clone()])
                elif 'bias' in k:
                    new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[i] += v[math.floor(float(i) / enlarge_n_times)]
                    lst.append([k, new_weight.clone()])
                else:
                    print('3')
                    print(k)
                    print('wrong')
            elif 'layer_norm' in k:
                if 'weight' in k:
                    new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[i] += v[math.floor(float(i) / enlarge_n_times)] * eff1
                    lst.append([k, new_weight.clone()])
                elif 'bias' in k:
                    new_weight = torch.empty((width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[i] += v[math.floor(float(i) / enlarge_n_times)] * eff1
                    lst.append([k, new_weight.clone()])
                else:
                    print('3')
                    print(k)
                    print('wrong')
            else:
                if 'weight' in k:
                    new_weight = torch.empty((emb_num, width * enlarge_n_times), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    for i in range(width * enlarge_n_times):
                        new_weight[: emb_num, i] += v[:, math.floor(float(i) / enlarge_n_times)] * eff1
                    lst.append([k, new_weight.clone()])
                elif 'bias' in k:
                    new_weight = torch.empty((emb_num), dtype = torch.float16, device = 'cuda')
                    nn.init.normal_(new_weight, std=0.01)
                    new_weight[: emb_num] += v
                    lst.append([k, new_weight.clone()])
                else:
                    print('4')
                    print(k)
                    print('wrong')
        else:
            print('5')
            print(k)
            print('wrong')
    new_lst = []
    if whether_stack:
        for k, v in lst:
            k_split = k.split('.')
            if k_split[0] == 'decoder' and k_split[2] == 'layers':
                l_id = int(k_split[3])
                k_split[3] = str(l_id + ckpt['args'].encoder_layers)
                new_k = '.'.join(k_split)
                new_lst.append([new_k, v.clone()])
        ckpt['args'].encoder_layers *= 2

    for k, v in lst:
        ckpt['model'][k] = v
    for k, v in new_lst:
        ckpt['model'][k] = v
    for k, v in ckpt['model'].items():
        print(k)
        print(v.size())
    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()
