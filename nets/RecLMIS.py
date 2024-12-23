# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .pixlevel import PixLevelModule
from .Interactor import Interactor
from .module_clip import CLIP, convert_weights, _PT_NAME
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, Slip
from .transformer import DualTransformer
from .transformer.mutihead_attention import MultiheadAttention
from .transformer.xpool import XPool
import time

# import clip


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class RecLMIS(nn.Module):
    def __init__(self, global_config, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.config = config
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.loss_weight = global_config.loss_weight
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.interact = Interactor(config, vis, img_size=14, channel_num=512, patch_size=1, embed_dim=512)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpblockAttention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpblockAttention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpblockAttention(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpblockAttention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss
        self.multi_activation = nn.Softmax()
        self.text_module4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.load_clip(config)
        self.mlp_woi =  nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1))
        self.mlp_visual =  nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1))

        self.rec_text_trans1 = DualTransformer(num_heads=4, num_decoder_layers1=self.config.rec_trans_num_layers1, num_decoder_layers2=self.config.rec_trans_num_layers1)
        self.rec_img_trans1 = DualTransformer(num_heads=4, num_decoder_layers1=self.config.rec_trans_num_layers1, num_decoder_layers2=self.config.rec_trans_num_layers1)

        self.mse_loss = nn.MSELoss(reduction='none')
        self.loss_fct = CrossEn(config)
        self.dropout1 = nn.Dropout(p=config.dropout_value)
        self.aux = True

        
    def load_clip(self, config):
        # Load the model
        backbone = config.clip_backbone
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        print("use clip version:", model_path)
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        print(transformer_width, transformer_heads, transformer_layers)
        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16
        self.clip.load_state_dict(state_dict, strict=False)
        self.clip.float()
        # print(list(self.clip.children()))
        # print(list(self.clip.parameters()))
        
        if self.config.frozen_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        # for name, param in self.clip.named_parameters():
        #     print(f"Parameter Name: {name}", end="-")
        #     print(f"Is Frozen: {not param.requires_grad}")

    def _mask_feat(self, feat, feat_len, weights=None, mask_rate = 0.3, mode='dist', mask_idx='1', mask_num=None):
        
        masked_vec = []
        for i, l in enumerate(feat_len):
            l = int(l)
            if mask_num is not None:
                num_masked_vec = max(int(mask_num), 1)
            else:
                num_masked_vec = max(int(l * mask_rate), 1) 
            masked_vec.append(torch.zeros([feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().detach().numpy() if weights is not None else None
            # choices = np.random.choice(np.arange(l), num_masked_vec, replace=False)
            if mode=='dist':
                if np.sum(p>0) <= num_masked_vec:
                    num_masked_vec_origin = num_masked_vec
                    num_masked_vec = np.sum(p>0)
                choices = np.random.choice(np.arange(l), num_masked_vec, replace=False, p=p)
                if np.sum(p>0) <= num_masked_vec and self.config.mask_mode_dist_random:
                    target_indices = np.arange(l)
                    target_indices = np.setdiff1d(target_indices, choices)

                    selected_index = np.random.choice(target_indices, num_masked_vec_origin-num_masked_vec, replace=False)

                    choices = np.append(choices, selected_index)

            elif mode=='topk':
                choices = torch.topk(weights[i, :l], k=num_masked_vec)[1]
            masked_vec[-1][choices] = 1

        masked_vec = torch.stack(masked_vec, 0).unsqueeze(-1)
        if mask_idx == '1':
            out_feat = feat.masked_fill(masked_vec == 1, 0)
        elif mask_idx == '0':
            out_feat = feat.masked_fill(masked_vec == 0, 0)

        return out_feat, masked_vec
    
    def reconstructor(self, text_feat4, text_feat, text_mask, img_feat4, img_feat, text_feat_clip):
        
        bsz,l,T = text_feat.shape
        img_feat4 = img_feat4.flatten(2).transpose(-1, -2)
        img_mask = torch.ones((bsz, img_feat4.shape[1])).to(text_mask.device)
        # b,n,d
        img_feat_inter = img_feat.clone()
        text_feat_inter = text_feat.clone()
        if self.config.dropout:
            img_feat = self.dropout1(img_feat)
        img_weight = self.mlp_visual(img_feat).squeeze(2)
        img_weight = img_weight.masked_fill_(torch.tensor((1 - img_mask.int()), dtype=torch.bool), float("-inf"))
        img_weight = torch.softmax(img_weight, dim=-1)  # B_t x N_t
        masked_img_feat, masked_img_index = self._mask_feat(img_feat4, img_mask.sum(1), weights=img_weight, mask_rate=self.config.img_mask_rate, mode=self.config.mask_mode)

        # [b,l,d] 
        if self.config.dropout:
            text_feat = self.dropout1(text_feat)
        text_weight = self.mlp_woi(text_feat).squeeze(2)
        text_weight = text_weight.masked_fill_(torch.tensor((1 - text_mask.int()), dtype=torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B_t x N_t
        masked_text_feat, masked_text_index = self._mask_feat(text_feat4, text_mask.sum(1), weights=text_weight, mask_rate=self.config.text_mask_rate, mode=self.config.mask_mode)

        img_rec_res = self.lambda_reconstructor(text_feat4, masked_img_feat, text_weight, img_mask, mode="img", hard_neg=True, neg=True)

        text_rec_res = self.lambda_reconstructor(img_feat4, masked_text_feat, img_weight, text_mask, mode="text", hard_neg=True, neg=True)

        loss_text_rec = self.mse_loss(text_rec_res, text_feat_clip).mean()

        loss_img_rec = self.mse_loss(img_rec_res, img_feat_inter).mean()

        if torch.any(torch.isnan(loss_text_rec)):
            loss_text_rec = torch.tensor(0).to(loss_text_rec.device).to(torch.float32)
        if torch.any(torch.isnan(loss_img_rec)):
            loss_img_rec = torch.tensor(0).to(loss_img_rec.device).to(torch.float32)

        return {"loss_text_rec": loss_text_rec,
                "loss_img_rec": loss_img_rec,
                }, img_weight, text_weight

    def lambda_reconstructor(self, scr1, scr2, weight, bool_mask, mode="text", hard_neg=True, neg=True):
        if mode=="text":
            rec_res = self.rec_text_trans1(scr1, None, scr2, None, decoding=2, gauss_weight=weight)[1]
        elif mode=="img":
            rec_res = self.rec_img_trans1(scr1, None, scr2, None, decoding=2, gauss_weight=weight)[1]

        return rec_res

    def cond_cons_loss(self, text_feat, text_mask, img_feat, text_weight=None, img_weight=None):
        img_feat = img_feat.flatten(2).transpose(-1,-2)
        text_weight = torch.softmax(text_weight, dim=-1)  # B_t x N_t
        img_weight = torch.softmax(img_weight, dim=-1)  # B_t x N_t

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        retrieve_logits1 = torch.einsum('atd,bvd->abtv', [text_feat, img_feat])
        retrieve_logits1 = torch.einsum('abtv,at->abtv', [retrieve_logits1, text_mask])

        t2v_logits, max_idx1 = retrieve_logits1.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
        v2t_logits, max_idx2 = retrieve_logits1.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, img_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        logit_scale = self.clip.logit_scale.exp()

        loss_t2v = self.loss_fct(retrieve_logits * logit_scale)
        loss_v2t = self.loss_fct(retrieve_logits.T * logit_scale)
        
        loss = (loss_t2v + loss_v2t) / 2

        return loss
        
    

    def forward(self, images, masks, text_token, text_mask, mode="train"):
        
        loss_dic = {}
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        cls, text_feat = self.clip.encode_text(text_token, return_hidden=True, mask=text_mask)
        # print(text_token, text_feat)
        text_feat_clip = text_feat.clone()
        # start = time.time()
        x = images.float()  
        x1 = self.inc(x)  
        
        text_feat = self.text_module4(text_feat.float().transpose(1, 2)).transpose(1, 2) #
        img_feat1 = x1 #### results are very volatile in this way.
        # b,64,224,224 --> b,128,112,112
        img_feat1 = self.down1(img_feat1)


        # b,128,112,112 --> b,256,56,56
        img_feat2 = self.down2(img_feat1)

        # b,256,56,56 --> b,512,28,28
        img_feat3 = self.down3(img_feat2)

        # b,512,28,28 --> b,512,14,14
        img_feat4 = self.down4(img_feat3)
        text_feat4_4rec, img_feat4_4rec = self.interact(img_feat4, text_feat)
        # self.aux = False
        if self.aux:
            try:
                rec_loss_dic, img_weight, text_weight = self.reconstructor(text_feat, text_feat4_4rec, text_mask, img_feat4, img_feat4_4rec, text_feat_clip)
                
                if self.loss_weight["loss_ccl"]!=0:
                    cond_cons_loss = self.cond_cons_loss(text_feat, text_mask, img_feat4, text_weight=text_weight, img_weight=img_weight)
                    loss_dic["loss_ccl"] = cond_cons_loss
                else:
                    loss_token_cons = 0
            except:
                self.aux = False
                rec_loss_dic = {}
        else:
            rec_loss_dic = {}

        x = self.up4(img_feat4_4rec.transpose(-1,-2).view(-1, 512, 14, 14), img_feat3)

        x = self.up3(x, img_feat2)
        x = self.up2(x, img_feat1)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            x = self.outc(x)
            # print("infer time: ", time.time()-start)
            logits = self.last_activation(x)
        else:
            logits = self.outc(x)  # if not using BCEWithLogitsLoss or class>1
        loss_dic.update(rec_loss_dic)
        if mode=="test":
            return logits, img_weight, text_weight
        else:
            return logits, loss_dic
