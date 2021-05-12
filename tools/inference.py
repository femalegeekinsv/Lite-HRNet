
import argparse
import os
import torch
import mmcv

from mmcv import Config
from mmpose.apis import init_pose_model, inference_top_down_pose_model

VALID_IMG_TYPES = ['jpg','jpeg', 'png']
def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img-path', help='Input Image for inference')
    parser.add_argument('--save', 
        choices=['none', 'image', 'heatmaps', 'keypoints', 'composite', 'all'],
        default='none',
        help='save output files')
    parser.add_argument('--display', action='store_true', help='whether to display composite output')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = 'cuda:0' if torch.cuda.is_available() else None

    model = init_pose_model(config=cfg, checkpoint=args.checkpoint, device=device)
    img_path = args.img_path
    
    if os.path.isfile(img_path):
        Exception("--img-path value is not a valid file path")
    elif lower(img_path.split('.')[-1]) not in VALID_IMG_TYPES:
        Exception(f"--img-path value is not a valid file type. \n Valid file types are {VALID_IMG_TYPES}")

    output = inference_top_down_pose_model(model, img_path)
    

if __name__ == '__main__':
    main()