import torch
import numpy as np

def dummy(x, y):
    return x

def compute_args(args):

    # DataLoader
    if args.dataset == "MOSEI": args.dataloader = 'Mosei_Dataset'
    if args.dataset == "MELD": args.dataloader = 'Meld_Dataset'
    if args.dataset == "PRE": args.dataloader = 'PretrainDataset'
    if args.dataset == "PRE_SIM": args.dataloader = 'Mosei_Dataset'

    # Loss function to use
    if args.dataset == 'MOSEI' and args.task == 'sentiment': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    if args.dataset == 'MOSEI' and args.task == 'emotion': args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    if args.dataset == 'MELD': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    if args.dataset == "PRE": args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    if args.dataset == "PRE_SIM": args.loss_fn = dummy

    # Answer size
    if args.dataset == 'MOSEI' and args.task == "sentiment": args.ans_size = 7
    if args.dataset == 'MOSEI' and args.task == "sentiment" and args.task_binary: args.ans_size = 2
    if args.dataset == 'MOSEI' and args.task == "emotion": args.ans_size = 6
    if args.dataset == 'MELD' and args.task == "emotion": args.ans_size = 7
    if args.dataset == 'MELD' and args.task == "sentiment": args.ans_size = 3
    if args.dataset == "PRE": args.ans_size = 2


    if args.dataset == 'MOSEI': args.pred_func = "amax"
    if args.dataset == "PRE": args.pred_func = "amax"
    if args.dataset == 'MOSEI' and args.task == "emotion": args.pred_func = "multi_label"
    if args.dataset == 'MELD': args.pred_func = "amax"

    return args