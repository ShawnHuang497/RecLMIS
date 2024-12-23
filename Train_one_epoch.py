# -*- coding: utf-8 -*-
import torch.optim
import os
import time
from utils import *
import Config_covid19 as config
import warnings
warnings.filterwarnings("ignore")


def print_summary(epoch, i, nb_batch, loss, loss_dict, loss_name, batch_time,
                  average_loss, average_time, iou, average_iou,
                  dice, average_dice, acc, average_acc, mode, lr, logger, data_time_ave):
    '''
        mode = Train or Test
    '''
    current_datetime = time.localtime()
    summary = '   [' + str(mode) + '] {}-{}-{} {}:{}:{} Epoch: [{}][{}/{}]  '.format(current_datetime.tm_year,
        current_datetime.tm_mon,
        current_datetime.tm_mday,
        current_datetime.tm_hour,
        current_datetime.tm_min,
        current_datetime.tm_sec,
        epoch, i, nb_batch)
    string = ''
    string += 'Back_Loss:{:.3f} '.format(loss)
    for k,v in loss_dict.items():
        string += '{}:{:.3f} '.format(k, v)
    string += '(Avg {:.4f}) '.format(average_loss)
    string += '|| IoU:{:.3f} '.format(iou)
    string += '(Avg {:.4f}) '.format(average_iou)
    string += 'Dice:{:.4f} '.format(dice)
    string += '(Avg {:.4f}) '.format(average_dice)
    # string += 'Acc:{:.3f} '.format(acc)
    # string += '(Avg {:.4f}) '.format(average_acc)
    if mode == 'Train':
        string += 'LR {:.6f}   '.format(lr)
    # string += 'Time {:.1f} '.format(batch_time)
    string += '(AvgTime {:.1f})   '.format(average_time)
    string += '(AvgDataTime {:.2f})   '.format(data_time_ave)
    summary += string
    logger.info(summary)
    # print summary


##################################################################################
#=================================================================================
#          Train One Epoch
#=================================================================================
##################################################################################
def train_one_epoch(config, loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, logger):
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum, data_time_sum = 0, 0, 0
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    dices = []
    dataiter = iter(loader)
    steps = len(loader)
    for i in range(1, steps+1):
        time0 = time.time()
        sampled_batch, names = next(dataiter)
        data_time = time.time() - time0
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images, masks, text_token, text_mask = sampled_batch["image"], sampled_batch["label"], sampled_batch["text_token"], sampled_batch["text_mask"]
        
        images, masks, text_token, text_mask = images.cuda(), masks.cuda(), text_token.cuda(), text_mask.cuda()


        # ====================================================
        #             Compute loss
        # ====================================================

        preds, loss_dict = model(images, masks, text_token, text_mask)
        loss_criterion = criterion(preds, masks.float())  # Loss
        # print(model.training)
        # print(out_loss, loss_sim)
        
        loss_dict['loss_criterion'] = loss_criterion

        out_loss = 0
        for k,v in loss_dict.items():
            loss_dict[k] = v.mean()
            out_loss += config.loss_weight[k] * v.mean()
        # out_loss = out_loss + loss_sim

        if model.training:
            optimizer.zero_grad()
            out_loss.backward()
            optimizer.step()

        train_dice = criterion._show_dice(preds, masks.float())
        train_iou = iou_on_batch(masks, preds)

        batch_time = time.time() - end
        if (epoch + 1) % config.vis_frequency == 0 and logging_mode == 'Val':
            vis_path = config.visualize_path+str(epoch)+'/'
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            save_on_batch(images,masks,preds,names,vis_path)
        dices.append(train_dice)

        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        iou_sum += len(images) * train_iou
        # acc_sum += len(images) * train_acc
        dice_sum += len(images) * train_dice
        data_time_sum += data_time

        if i == len(loader):
            average_loss = loss_sum / (config.batch_size*(i-1) + len(images))
            average_time = time_sum / (config.batch_size*(i-1) + len(images))
            train_iou_average = iou_sum / (config.batch_size*(i-1) + len(images))
            # train_acc_average = acc_sum / (config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (config.batch_size*(i-1) + len(images))
        else:
            average_loss = loss_sum / (i * config.batch_size)
            average_time = time_sum / (i * config.batch_size)
            train_iou_average = iou_sum / (i * config.batch_size)
            # train_acc_average = acc_sum / (i * config.batch_size)
            train_dice_avg = dice_sum / (i * config.batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % config.print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), out_loss, loss_dict, loss_name, batch_time,
                          average_loss, average_time, train_iou, train_iou_average,
                          train_dice, train_dice_avg, 0, 0,  logging_mode,
                          lr=min(g["lr"] for g in optimizer.param_groups),logger=logger, 
                          data_time_ave=data_time_sum/config.print_frequency)
            data_time_sum = 0
        if config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_iou', train_iou, step)
            # writer.add_scalar(logging_mode + '_acc', train_acc, step)
            writer.add_scalar(logging_mode + '_dice', train_dice, step)
        if config.lr == 'poly' and lr_scheduler is not None:
            lr_scheduler.step()

        torch.cuda.empty_cache()

    # if config.lr != 'poly':
    #     lr_scheduler.step()
    if lr_scheduler is not None and config.lr != 'poly':
        lr_scheduler.step()

    return average_loss, train_dice_avg
