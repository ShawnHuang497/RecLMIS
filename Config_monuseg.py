# -*- coding: utf-8 -*-
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
use_cuda = torch.cuda.is_available()
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)

lr = 'cosineLR'  # Use cosineLR, exp, cosine, poly
n_channels = 3
n_labels = 1  
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 5000
early_stopping_patience = 50

pretrain = False
task_name = 'MoNuSeg' 
token_len = 10
learning_rate = 1e-3 


batch_size = 24

optimizer = "Adam"
weight_decay = 1e-5

model_name = 'RecLMIS'
loss_weight = { "loss_criterion": 5, 
                "loss_ccl": 0.2, 
                "loss_text_rec": 1,
                "loss_img_rec": 1,
                }
text_name = "text_alpha.xlsx"
resume = False

train_dataset = './datasets/' + task_name + '/Train_Folder/'
val_dataset = './datasets/' + task_name + '/Val_Folder/'
test_dataset = './datasets/' + task_name + '/Test_Folder/'


session_name = 'session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'

##########################################################################
# ViT configs
##########################################################################
def get_ViT_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.base_channel = 64 
    config.clip_backbone = "ViT-B/32"
    config.text_mask_rate = 0.3
    config.img_mask_rate = 0.5
    config.pool_mode = "max_pool"  # max_pool, aver_pool
    config.rec_trans_num_layers1 = 3
    config.mask_mode = "dist"
    config.frozen_clip = True
    config.mask_mode_dist_random = True
    config.dropout = True
    config.dropout_value = 0.5
    return config


# used in testing phase, copy the session name in training phase
test_session = "session_08.20_20h40" 
test_vis = False
