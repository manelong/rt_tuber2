'''导入训练好的模型，进行推理，存储中间结果，用于后续的评估'''

import os 
import sys 

import numpy as np  
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn 

from src.core import YAMLConfig

def save_outputs(labels, bboxes, save_path):
    """将每次输出的outputs按指定的格式储存在pkl文件下
    """
    labels = labels.cpu().numpy()

    for i in range(len(labels)):
        bboxes[i] = bboxes[i].cpu().numpy()



    pass


class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
            
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # 载入模型参数
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    model = Model(cfg)
    model.eval()
    model.to(args.device)

    val_dataloader = cfg.val_dataloader
    # train_dataloader = cfg.train_dataloader

    for i, (samples, targets) in enumerate(val_dataloader):
        fast_pathway = samples.to(args.device)
        slow_pathway = torch.index_select(
            samples,
            2,
            torch.linspace(
                0, samples.shape[2] - 1, samples.shape[2] // 8
            ).long(),
        ).to(args.device)
        samples = [slow_pathway, fast_pathway]
        size = torch.tensor([[320, 240]]).to(args.device)
        outputs = model(samples, size)
        
        out_files = []
        all_pred = []
        for j in range(len(targets)):
            out_files.append(targets[j]['image_path'].replace('.jpg', '.pkl').replace('rgb-images', 'outputs'))

            outputs_labels = outputs[0][j]
            outputs_bboxes = outputs[1][j]
            outputs_confids = outputs[2][j]

            top_pred_bbox = {}
            top_pred = {}
            for k in range(24):
                # 找到在outputs中与i类别相同的所有index
                index = torch.where(outputs_labels == k)
                top_pred_bbox[k] = outputs_bboxes[index].detach().cpu().numpy()
                # 将对应的置信度也存储下来，存在top_pred中
                top_pred[k] = []
                for w in range(len(top_pred_bbox[k])):
                    # 希望top_pred[k][w]的后面加上置信度
                    top_pred[k].append(np.append(top_pred_bbox[k][w], outputs_confids[index][w].detach().cpu().numpy()))
                top_pred[k] = np.array(top_pred[k])
            # top_pred转为np数组储存
            all_pred.append(top_pred)
            
        print('model: ', outputs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/rtdetrv2/tube_test.yml')
    parser.add_argument('--resume', '-r', type=str, default='output/tube_det_1/last.pth')
    parser.add_argument('--device', '-d', type=str, default='cuda')

    args = parser.parse_args()

    main(args)
