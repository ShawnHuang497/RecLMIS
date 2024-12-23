# -*- coding: utf-8 -*-

import argparse
import os
# import Config_covid19 as config
parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', '-c', default='Config_covid19', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file')
parser.add_argument('--gpu', '-g', default='0', metavar='cuda',
                    type=str,
                    help='device id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cfg_path == "Config_monuseg":
    import Config_monuseg as config
elif args.cfg_path == "Config_Kvasir_Clinic":
    import Config_MosMedPlus as config
else:
    import Config_covid19 as config

import torch.optim
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch.backends import cudnn
# import Config
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D
from nets.RecLMIS import RecLMIS
from torch.utils.data import DataLoader
import logging
from Train_one_epoch import train_one_epoch, print_summary
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, WeightedDiceCE, read_text, read_text_LV, save_on_batch
from thop import profile

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '' + \
                   'latest_model.pth.tar'
    logger.info('\t Saving to {}'.format(filename))
    torch.save(state, filename)


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


##################################################################################
# =================================================================================
#          Main Loop: load model,
# =================================================================================
##################################################################################
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
    # Load train and val data
    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    if config.task_name == 'MoNuSeg' or config.task_name == 'MosMedplus':
        train_text = read_text(config.train_dataset + 'Train_text.xlsx')
        val_text = read_text(config.val_dataset + 'Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, train_text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="val")
    elif config.task_name == 'Covid19':
        text = read_text(config.task_dataset + 'Train_Val_text.xlsx')
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size, data_name=config.task_name, config=config, mode="val")
    elif config.task_name == 'Kvasir_Clinic':
        # text = read_text(config.val_dataset + '../{}'.format(config.text_name))

        text = read_text(config.train_dataset + '{}'.format(config.text_name))
        
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, text, val_tf, image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="val")
    elif config.task_name == 'Kvasir_Clinic_Pra':
        text = read_text(config.train_dataset + '{}'.format(config.text_name))
        print('val_text: ',config.val_dataset + '{}'.format(config.text_name))
        val_text = read_text(config.val_dataset + '{}'.format(config.text_name))
        train_dataset = ImageToImage2D(config.train_dataset, config.task_name, text, train_tf,
                                       image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="train")
        val_dataset = ImageToImage2D(config.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size, data_name=config.task_name, token_len=config.token_len, config=config, mode="val")

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)
                             
    lr = config.learning_rate
    logger.info(model_type)


    config_vit = config.get_ViT_config()
    model = RecLMIS(config, config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


    input = torch.randn(2, 3, 224, 224)
    text = torch.randn(2, 10, 768)
    # flops, params = profile(model, inputs=(input, text, text))
    # logger.info('flops:{}'.format(flops))
    # logger.info('params:{}'.format(params))
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize
    if config.lr == 'cosineLR':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    elif config.lr == 'exp':
        lambda1 = lambda epoch: max(0.99**epoch, 0.1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
    elif config.lr == 'cosine':
        warm_up_steps = 0
        warm_up_with_cosine_lr = lambda step: step / warm_up_steps if step <= warm_up_steps and warm_up_steps!=0 else 0.5 * (math.cos((step - warm_up_steps) /(config.epochs - warm_up_steps) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif config.lr == 'poly':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lambda x: (1 - x / (len(train_loader) * config.epochs)) ** 0.99)

    print(config.lr)
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    epoch = 0

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        # print(type(checkpoint), type(checkpoint['model']), checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    model = model.cuda()

    if config.resume:
        checkpoint = torch.load(config.resume_path, map_location='cpu')
        logger.info('resume path: {}'.format(config.resume_path))
        print(model.load_state_dict(checkpoint['state_dict']))
        
    if torch.cuda.device_count() > 1:
        logger.info("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    if config.resume:
        print(optimizer.load_state_dict(checkpoint['optimizer']))
        print(lr_scheduler.load_state_dict(checkpoint['lr_scheduler']))
        epoch = checkpoint['epoch']
        print("resume optimizer and lr scheduler successfuly")
    else:
        epoch = -999

    max_dice = 0.0
    for epoch in range(max(0, epoch+1), config.epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(config, train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)  # sup

        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(config, val_loader, model, criterion,
                                                 optimizer, writer, epoch, lr_scheduler, model_type, logger)
        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
            if epoch + 1 > 0:
                logger.info(
                    '\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice, val_dice))
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice, max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        save_checkpoint({'epoch': epoch,
                        'best_model': False,
                        'model': model_type,
                        'state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()}, config.model_path)

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':

    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    logger = logger_config(log_path=config.logger_path)

    with open(args.cfg_path+'.py', 'r') as file:  
        lines = file.readlines()  
    for line in lines:  
        logger.info(line[:-1])

    model = main_loop(model_type=config.model_name, tensorboard=True)
