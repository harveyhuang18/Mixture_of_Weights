import copy
from torch import nn
import re
import torch
from tqdm import tqdm
from transformers import MobileNetV2ForImageClassification
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, random_split
import os
import pickle
import math
import numpy as np


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def torch_load_old(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model



def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, 'to'):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, 'predict_proba'):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Gating_Net(nn.Module):#Input size:3, 224, 224
    def __init__(self, base_ckpt='google/mobilenet_v2_1.0_224', num_task=8):
        super().__init__()
        base_model = MobileNetV2ForImageClassification.from_pretrained(base_ckpt)
        self.model = base_model
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=num_task),
            nn.Softmax(dim=1)
        )
        self.num_task = num_task
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        return self.model(x)

    def test_gating(self, ds, batch_size=32, device='cpu'):
        valid_epoch_acc = 0
        valid_epoch_loss = 0
        self.model.eval()
        self.model.to(device)
        valid_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
        for x, y in tqdm(valid_loader, desc='Validating Gating'):
            loss = 0
            acc = 0
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                output = self.model(x)
                loss = loss + self.criterion(output.logits, y)
                acc += (output.logits.argmax(dim=1) == y).float().mean()
                valid_epoch_loss += loss / len(valid_loader)
                valid_epoch_acc += acc / len(valid_loader)
        return valid_epoch_loss, valid_epoch_acc


    def train_gating_one_epoch(self, ds, batch_size=32, device='cpu', lr=1e-5):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_epoch_acc = 0
        train_epoch_loss = 0
        self.model.train()
        self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad=True
        train_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
        for x, y in tqdm(train_loader, desc='Training Gating'):
            self.optimizer.zero_grad()
            loss = 0
            acc = 0
            x = x.to(device)
            y = y.to(device)
            output = self.model(x)
            loss = loss + self.criterion(output.logits, y)
            acc += (output.logits.argmax(dim=1) == y).float().mean()
            loss.backward()
            self.optimizer.step()
            train_epoch_loss += loss / len(train_loader)
            train_epoch_acc += acc / len(train_loader)
        return train_epoch_loss, train_epoch_acc


class Gating_dataset(Dataset):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = copy.deepcopy(datasets)
        self.min_len = 100
        self.lens = [max(len(self.datasets[i]), self.min_len) for i in range(len(self.datasets))]

    def __len__(self):
        length = 0
        for ds in self.datasets:
            length += max(self.min_len, len(ds))
        return length

    def __getitem__(self, idx):
        if idx < self.lens[0]:
            DATA = self.datasets[0][idx]
            Y = 0
        else:
            idx -= self.lens[0]
            for i in range(1, len(self.datasets)):
                if self.lens[i] > idx:
                    DATA = self.datasets[i][(idx) % len(self.datasets[i])]
                    Y = i
                    break
                else:
                    idx -= self.lens[i]
        return DATA[0], Y

    def split_train_ds(self, train_ratio):
        ds_gating_train, _ = random_split(self, lengths=[int(len(self) * train_ratio), len(self) - int(len(self) * train_ratio)])
        return ds_gating_train


def get_params(models, filter=None, exclude=None):
    params = {}
    for model in models:
        n2p = model#{k: v for k,v in model.named_parameters()}
        merge_param_names = filter_params_to_merge([n for n in n2p], [exclude])
        for n in merge_param_names:
            if filter is None:
                if n not in params:
                    params[n] = []
                params[n].append(n2p[n])
            else:
                if len(filter) == 1:
                    if filter[0] in n:
                        if n not in params:
                            params[n] = []
                        params[n].append(n2p[n])
                else:
                    for f in filter:
                        if f in n:
                            if n not in params:
                                params[n] = []
                            params[n].append(n2p[n])
    return params

def filter_params_to_merge(param_names, exclude_param_regex):
    params_to_merge = []
    for name in param_names:
        if exclude_param_regex == [None]:
            valid = True
        else:
            valid = not any([re.match(patt, name) for patt in exclude_param_regex])
        if valid:
            params_to_merge.append(name)
    return params_to_merge

def compute_sim_params_concat(params, method, args):
    SIM = []
    if args.model =='ViT-L-14':
        num_block = 24
    else:
        num_block = 12
    for block in range(num_block):
        p = {}
        for n in params:
            if '.{}.'.format(block) in n:
                p[n] = params[n]
        cat = []
        for n in p:
            for i in range(len(p[n])):
                if len(cat) < i+1:
                    cat.append( torch.flatten(p[n][i]))
                else:
                    cat[i] = torch.concat([cat[i], torch.flatten(p[n][i])])

        sim = 0
        num = 0
        for i in range(len(cat)):
            for j in range(i + 1, len(cat)):
                num += 1
                if method == 'cossim':
                    sim += torch.cosine_similarity(cat[i], cat[j], dim=0)
                elif method == 'l1':
                    sim += 1 - torch.nn.L1Loss()(cat[i], cat[j])
                elif method == 'l2':
                    sim += 1 - torch.nn.MSELoss()(cat[i], cat[j])
                else:
                    raise ValueError('METHOD ERROR!')

        SIM.append(sim / num)
    return SIM

def sim_FFN_cat(models, params=['mlp.c'], metric='cossim', args=None):
    with torch.no_grad():
        param_ffn = get_params(models,params, exclude='.*attention.*')
        SIM = compute_sim_params_concat(param_ffn, metric, args)
    sorted_id = sorted(range(len(SIM)), key=lambda k: SIM[k], reverse=False)
    selected_param = {}
    for i in range(args.hyper_param_m):
        selected_id = sorted_id[i]
        for n in param_ffn:
            if '.{}.'.format(selected_id) in n:
                selected_param[n] = param_ffn[n]
    return SIM, selected_param, sorted_id


def sim_attn_cat(models, params=['lora'], metric='cossim', args=None):
    with torch.no_grad():
        param_ffn = get_params(models, params)
        sim = compute_sim_params_concat(param_ffn, metric, args)

    sorted_index = sorted(range(len(sim)), key=lambda k: sim[k], reverse=False)
    selected_param = {}
    for i in range(args.hyper_param_m):
        selected_id = sorted_index[i]
        for n in param_ffn:
            if '.{}.'.format(selected_id) in n:
                selected_param[n] = param_ffn[n]
    return sim, selected_param, sorted_index


def ln_embed_param(params):
    param_ln_embed = get_params(params, ['embed','ln_1', 'ln_2'])
    return param_ln_embed


def gating_merge_param(params, ratio):
    merged = {}
    ratio = ratio.to('cpu')
    for n in params:
        for i in range(len(params[n])):
            if n not in merged:
                merged[n] = params[n][i] * ratio[0, i]
            else:
                merged[n] += params[n][i] * ratio[0, i]
    return merged


