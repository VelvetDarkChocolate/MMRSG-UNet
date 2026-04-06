import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, powerset
from torchvision import transforms
import numpy as np
from scipy.ndimage import zoom

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

def trainer_synapse(args, model, snapshot_path, optimizer_params=None):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    

    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        try:
            root_logger.removeHandler(h)
            h.close()
        except Exception:
            pass
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    root_logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu


    train_list_dir = getattr(args, 'list_dir_train', args.list_dir)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=train_list_dir, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    logging.info("The length of train set is: {}".format(len(db_train)))

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    if optimizer_params is None:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    else:
        optimizer = optim.AdamW(optimizer_params, weight_decay=0.0001)
        
    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))
    start_epoch = getattr(args, 'start_epoch', 0)
    iter_num = start_epoch * len(trainloader)
    max_epoch = args.max_epochs

 
    out_idxs = list(np.arange(4))
    ss = [x for x in powerset(out_idxs) if x]
    logging.info("Using Combinatorial Mutation Loss, subsets count: {}".format(len(ss)))

    best_performance = 0.0
    best_epoch = -1

    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            outputs_list = model(image_batch)
            loss = 0.0
            w_ce, w_dice = 0.4, 0.6
            
            for s in ss:
                iout = 0.0
                for idx in s:
                    iout += outputs_list[idx]
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1
            
            writer.add_scalar('info/total_loss', loss, iter_num)
            loss_ce_main = ce_loss(outputs_list[0], label_batch[:].long())
            writer.add_scalar('info/loss_ce_main', loss_ce_main, iter_num)

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs_list[0], dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

    
        current_epoch = epoch_num + 1
        if current_epoch >= args.save_start_epoch and (current_epoch - args.save_start_epoch) % args.save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Saved model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("Saved final model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
