import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import MMRSGUNet as ViT_seg
from trainer import trainer_synapse
from config import get_config
'''
python train.py \
  --cfg configs/cswin_tiny_224_lite.yaml \
  --output_dir your OUT_DIR \
  --root_path  your DATA_DIR \
  --max_epochs 250 \
  --batch_size 24 \
  --base_lr 0.0001
'''
parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str, default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, required=True, help='output dir')                   
parser.add_argument('--max_epochs', type=int, default=250, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument("--opts", help="Modify config options", default=None, nargs='+')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--save_start_epoch', type=int, default=120, help='start saving checkpoints from this epoch')
parser.add_argument('--save_interval', type=int, default=10, help='save checkpoint every N epochs')
parser.add_argument('--eval_interval', type=int, default=25, help='also save checkpoint every N epochs for evaluation')


parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='cache mode')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    net.load_from(config)
    
    if args.resume:
        msg = net.load_state_dict(torch.load(args.resume))
        logging.info(f"resume from {args.resume}: {msg}")
        try:
            base = os.path.basename(args.resume)
            if base.startswith('epoch_') and base.endswith('.pth'):
                args.start_epoch = int(base.replace('epoch_', '').replace('.pth', '')) + 1
        except Exception:
            args.start_epoch = 0

    logging.info("Setting differential learning rates...")
    new_module_params = list(net.mmrsg_unet.stage4.parameters()) + \
                        list(net.mmrsg_unet.stage_up4.parameters()) + \
                        list(net.mmrsg_unet.carafe4.parameters()) + \
                        list(net.mmrsg_unet.csgat4.parameters())
    new_module_param_ids = {id(p) for p in new_module_params}
    
    all_params = list(net.mmrsg_unet.parameters())
    old_module_params = [p for p in all_params if id(p) not in new_module_param_ids]

    optimizer_params = [
        {"params": new_module_params, "lr": args.base_lr * 10},
        {"params": old_module_params, "lr": args.base_lr}
    ]
    logging.info(f"New modules (Mamba bottleneck) parameters (LR={args.base_lr * 10}): {len(new_module_params)} tensors")
    logging.info(f"Pre-trained modules parameters (LR={args.base_lr}): {len(old_module_params)} tensors")

    trainer = {'Synapse': trainer_synapse}
    args.list_dir_train = args.list_dir
    args.list_dir_val = None
    trainer[args.dataset](args, net, args.output_dir, optimizer_params)
