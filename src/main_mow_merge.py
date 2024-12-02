from modeling import ImageClassifier
from heads import get_classification_head
import copy
from utils import *
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import sys

from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments
from torch import nn
from transformers import MobileNetV2ForImageClassification
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torchvision.transforms import TrivialAugmentWide
from datasets.registry import get_dataset, random_split

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
    model = 'ViT-B-32'
    args = parse_arguments()
    args.home = 'HOME-PATH'
    sys.path.append(args.home)
    args.data_location = args.home + '/data'
    args.model = model
    args.save = args.home + '/checkpoints/' + model
    args.logs_path = args.home + '/logs/' + model
    args.batch_size = 32
    pretrained_checkpoint = args.home + '/checkpoints/'+model+'/zeroshot.pt'
    args.device = 'cuda:0'
    args.hyper_param_m = 5
    args.pretrained_gating = None # Type the ckpt name if you have a trained gating net
    args.gating_training_epochs = 5
    args.gating_train_ratio = 0.03
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    log = create_log_dir(args.logs_path, 'log_{}_mow_merging.txt'.format(str_time_))
    pretrained_model = torch.load(pretrained_checkpoint)

    dss = []
    pps = copy.deepcopy(pretrained_model.val_preprocess) # Data Augmentation
    aug = TrivialAugmentWide()
    pps.transforms.insert(0, aug)

    for dataset_name in tqdm(exam_datasets, desc='loading dataset'):
        dss.append(get_dataset(dataset_name, pps, location=args.data_location, batch_size=args.batch_size).test_dataset)

    ds_gating = Gating_dataset(dss)
    ds_gating_train = ds_gating.split_train_ds(args.gating_train_ratio)

    if args.pretrained_gating is not None:
        gating = torch.load(args.home +'/gating_ckpt/'+args.pretrained_gating)
    else:
        gating = Gating_Net(base_ckpt='google/mobilenet_v2_1.0_224', num_task=len(exam_datasets))
        for epoch in tqdm(range(args.gating_training_epochs)):
            train_epoch_loss, train_epoch_acc = gating.train_gating_one_epoch(ds_gating_train, args.batch_size,
                                                                              args.device, 1e-4)
        torch.save(gating, args.home + f'/gating_ckpt/gating_{str_time_}.pth')



    ft_checks = [torch.load(args.home + '/checkpoints/' + args.model + '/' + dataset_name + '/finetuned.pt').state_dict() for
                 dataset_name in exam_datasets]

    log.info(f'---------lw_ada_pp w/ MoW, m={args.hyper_param_m}-------------')
    SIM_ATTN, selected_attn, _ = sim_attn_cat(ft_checks, ['attn'], args=args)
    SIM_FFN, selected_ffn1, _ = sim_FFN_cat(ft_checks, ['mlp'],  args=args)
    ln_embed = ln_embed_param(ft_checks)
    selected_param = dict(list(selected_attn.items()) + list(selected_ffn1.items()) + list(ln_embed.items()))
    adamerging_param = torch.load(args.home + '/checkpoints/' + args.model + '/lw_adamergingpp.pth', map_location=args.device)

    for i, [_, p] in enumerate(pretrained_model.named_parameters()):
        p.data.copy_(adamerging_param[i])
        print('Initializing!')

    classification_heads = [get_classification_head(args, exam_dataset) for exam_dataset in exam_datasets]
    accs = []
    gating.eval()
    for n, p in gating.named_parameters():
        p.requires_grad = False
    for dataset_name in exam_datasets:
        dataset = get_dataset(
            dataset_name,
            pretrained_model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )
        dataloader = dataset.test_loader
        corr = 0
        total = 0
        merged_vit = copy.deepcopy(pretrained_model)
        with torch.no_grad():
            for data in (pbar := tqdm(dataloader, desc="Evaluating", leave=False, dynamic_ncols=True)):
                x = data[0].to(args.device)
                y = data[1].to(args.device)
                gating_out = gating(x).logits
                gating_out = torch.mean(gating_out, dim=0, keepdim=True)
                pred_task_num = gating_out.argmax(dim=1)
                classification_head = classification_heads[pred_task_num]# Auto Head Selection
                gating_merged_param = gating_merge_param(selected_param, gating_out)
                for n, p in merged_vit.named_parameters():
                    if n in gating_merged_param:
                        p.data.copy_(gating_merged_param[n])
                merged_classifier = ImageClassifier(merged_vit, classification_head).to(args.device)
                out = merged_classifier(x)
                pred = out.argmax(dim=1, keepdim=True).to(args.device)
                corr += pred.eq(y.view_as(pred)).sum().item()
                total += y.size(0)
                pbar.set_postfix({"accuracy": corr / total})
            log.info("Dataset:{}; ACC:{}".format(dataset_name, corr / total))
            accs.append(corr / total)
    log.info(f'AVG:{np.mean(accs)}')