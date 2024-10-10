"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        fast_pathway = samples.to(device)
        slow_pathway = torch.index_select(
            samples,
            2,
            torch.linspace(
                0, samples.shape[2] - 1, samples.shape[2] // 8
            ).long(),
        ).to(device)
        samples = [slow_pathway, fast_pathway]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            # print('outputs:',outputs['pred_boxes'])
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def my_evaluate(model: torch.nn.Module, data_loader, device):
    # writer :SummaryWriter = kwargs.get('writer', None)

    model.eval()
    # criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'
    
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # global_step = epoch * len(data_loader) + i
        # metas = dict(epoch=epoch, step=i, global_step=global_step)

        with torch.autocast(device_type=str(device)):
            outputs = model(samples)
        
        # loss_dict = criterion(outputs, targets, **metas)
            
        # loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        # loss_value = sum(loss_dict_reduced.values())

        # metric_logger.update(loss=loss_value, **loss_dict_reduced)
        
    # stats = {}
    # if writer and dist_utils.is_main_process():
    #     writer.add_scalar('Val_Loss/total', loss_value.item(), epoch)
    #     for k, v in loss_dict_reduced.items():
    #         stats[k] = (v.item(), )
    #         writer.add_scalar(f'Val/{k}', v.item(), epoch)
        

    return 0

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, epoch, data_loader, device ,**kwargs):
    writer :SummaryWriter = kwargs.get('writer', None)

    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val:'
    
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        with torch.autocast(device_type=str(device)):
            outputs = model(samples)
        
        loss_dict = criterion(outputs, targets, **metas)
            
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        
    stats = {}
    if writer and dist_utils.is_main_process():
        writer.add_scalar('Val_Loss/total', loss_value.item(), epoch)
        for k, v in loss_dict_reduced.items():
            stats[k] = (v.item(), )
            writer.add_scalar(f'Val/{k}', v.item(), epoch)
        

    return stats


