
import os
import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets.RecLMIS import RecLMIS
from utils import *
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--cfg_path', '-c', default='Config_covid19', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file')
parser.add_argument('--gpu', '-g', default='0', metavar='cuda',
                    type=str,
                    help='device id')
parser.add_argument('--test_session', '-t', default='session_09.25_00h27',
                    type=str,
                    help='session name')
parser.add_argument('--test_vis', '-v', default=False, type=bool, help='visilization')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.cfg_path == "Config_monuseg":
    import Config_monuseg as config
elif args.cfg_path == "Config_Kvasir_Clinic":
    import Config_MosMedPlus as config
else:
    import Config_covid19 as config


red_color = (255, 0, 0)     # red
blue_color = (0, 0, 255)  # blue
green_color = (0, 255, 0)   # green
size = (224, 224)


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    return dice_pred, iou_pred


def pred_mix(ground_truth, prediction_mask, original_image):

    TP = np.sum(np.logical_and(ground_truth == 1, prediction_mask == 1))
    FP = np.sum(np.logical_and(ground_truth == 0, prediction_mask == 1))
    FN = np.sum(np.logical_and(ground_truth == 1, prediction_mask == 0))
    overlay = original_image.copy()

    # FN: red
    overlay[ground_truth == 1] = red_color
    # FP: blue
    overlay[np.logical_and(ground_truth == 0, prediction_mask == 1)] = blue_color
    # TP: green
    overlay[np.logical_and(ground_truth == 1, prediction_mask == 1)] = green_color
    
    return overlay

def draw_sub_plot(img, fig, nums, idx, mode="gray"):
    img = cv2.resize(img, size)
    ax = fig.add_subplot(1, nums, idx)
    if mode == "gray":
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.axis('off')
    

def vis_and_save_heatmap(model, input_img, masks, text_token, text_mask, img_RGB, labs, vis_save_path, dice_pred, dice_ens, text=None, config=config):
    model.eval()

    output, img_weight, text_weight = model(input_img, masks, text_token, text_mask, mode="test")
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
                                                  save_path=vis_save_path + '_predict' + model_type + '.jpg')


    original_image = torch.squeeze(input_img, 0).cpu().numpy() * 255
    original_image = original_image.transpose(1, 2, 0).astype(np.uint8)

    if args.test_vis:
        nums = 1
        fig = plt.figure(figsize=(nums,1), dpi=size[0])
        fig.subplots_adjust(wspace=0.01, left=0, right=1, bottom=0,top=1)
        # draw img
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        draw_sub_plot(original_image, fig, nums, 1, mode="rgb")
        original_image_ = original_image.copy()

        ## draw GT
        ground_truth = labs.squeeze()
        original_image_[ground_truth == 1] = green_color
        draw_sub_plot(original_image_, fig, nums, 2)

        # draw pred
        pred = pred_mix(ground_truth, predict_save, original_image)
        draw_sub_plot(pred, fig, nums, 3)
        
        fig.subplots_adjust(wspace=0.01, left=0, right=1, bottom=0, top=1)

        f = plt.gcf() 
        f.savefig(vis_save_path+"_dice"+str(round(dice_pred_tmp,2))+".png")
        f.clear()  

    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    test_session = args.test_session

    if config.task_name == "Covid19":
        model_type = config.model_name
        model_path = "./Covid19/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    elif config.task_name == 'MoNuSeg':
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    elif config.task_name == 'MosMedplus':
        model_type = config.model_name
        model_path = "./MosMedplus/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
    else:
        raise TypeError('Please enter a valid name for the task name in Config_xxx.py')
        
    
    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = save_path + config.task_name + '_visualize_test/'
    print("vis path is", vis_path)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'RecLMIS':
        config_vit = config.get_ViT_config()
        model = RecLMIS(config, config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    load_res = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('missing keys---> ', load_res.missing_keys)
    print('*'* 100)
    print('unexpected keys---> ', load_res.unexpected_keys)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_dataset + 'Test_text.xlsx')
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, 
                                    test_text, tf_test, image_size=config.img_size, 
                                    data_name=config.task_name, token_len=config.token_len, 
                                    config=config, mode="val")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0
    test_num = len(test_loader)
    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True, dynamic_ncols=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # Take variable and put them to GPU
            test_data, test_label, text_token, text_mask, text = sampled_batch["image"], sampled_batch["label"], sampled_batch["text_token"], sampled_batch["text_mask"], sampled_batch["text"]
            lab = test_label.data.numpy()

            test_data, test_label, text_token, text_mask = test_data.cuda(), test_label.cuda(), text_token.cuda(), text_mask.cuda()
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, test_data, test_label, text_token, text_mask, None, lab,
                                                           vis_path + str(names[0]), dice_pred=dice_pred, dice_ens=dice_ens, 
                                                           text=text, config=config)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.set_postfix({"dice_pred": round(dice_pred/i,4), "iou_pred": round(iou_pred/i, 4)})
            pbar.update()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
