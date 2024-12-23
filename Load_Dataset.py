# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
# from bert_embedding import BertEmbedding
import clip

from tools.transform import BioMedicalGaussianBlur, PhotoMetricDistortion
# from tools.tokenization_clip import SimpleTokenizer as ClipTokenizer

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text_token, text_mask = sample["image"], sample["label"], sample["text_token"], sample["text_mask"]
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        # text_mask = torch.Tensor(text_mask)
        sample = {'image': image, 'label': label, 'text_token': text_token, 'text_mask': text_mask}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text_token, text_mask = sample["image"], sample["label"], sample["text_token"], sample["text_mask"]
        image, label = image.astype(np.uint8), label.astype(np.uint8)  # OSIC
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)       
        # text_mask = torch.Tensor(text_mask)
        # sample = {'image': image, 'label': label, 'text_token': text_token, 'text_mask': text_mask}
        sample['image'] = image
        sample['label'] = label
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class LV2D(Dataset):
    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.output_path = os.path.join(dataset_path)
        self.mask_list = os.listdir(self.output_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.bert_embedding = BertEmbedding()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.output_path))

    def __getitem__(self, idx):

        mask_filename = self.mask_list[idx]  # Co
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1
        mask = correct_dims(mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        text_token = self.bert_embedding(text)
        text = np.array(text_token[0][1])
        if text.shape[0] > 14:
            text = text[:14, :]
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        sample = {'label': mask, 'text': text}

        return sample, mask_filename


class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224, data_name='Kvasir_Clinic', token_len=18, config=None, mode="train") -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        if data_name=="Kvasir_Clinic":
            self.output_path = os.path.join(dataset_path, '../bin_masks')
            self.images_list = os.listdir(self.input_path)
        else:
            self.output_path = os.path.join(dataset_path, 'labelcol')
            self.mask_list = os.listdir(self.output_path)
            self.images_list = os.listdir(os.path.join(dataset_path, 'img'))
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.data_name = data_name
        self.token_len = token_len
        self.config = config
        self.mode = mode
        self.BioMedicalGaussianBlur = BioMedicalGaussianBlur(prob=0.5)
        self.PhotoMetricDistortion = PhotoMetricDistortion()

        # self.bert_embedding = BertEmbedding()


        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def _get_text(self, video_id, sub_id):
        text_ids = self.tokenizer.convert_tokens_to_ids(words)

        return text_ids

    def __getitem__(self, idx):

        # image_filename = self.images_list[idx]  # MoNuSeg
        # mask_filename = image_filename[: -3] + "png"  # MoNuSeg
        if self.data_name == "Kvasir_Clinic" or self.data_name == "MoNuSeg" or self.data_name == 'MosMedplus':
            image_filename = self.images_list[idx]  
            mask_filename = image_filename[: -3] + "png"  
            # print(os.path.join(self.input_path, image_filename), os.path.join(self.output_path, mask_filename))
        else:
            mask_filename = self.mask_list[idx]  # Covid19
            image_filename = mask_filename
        if self.data_name == "Covid19":
            image_filename = mask_filename.replace('mask_', '')  # Covid19
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        try:
            image = cv2.resize(image, (self.image_size, self.image_size))
        except:

            print(os.path.join(self.input_path, image_filename))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')
        # text_token = self.bert_embedding(text)
        # text = np.array(text_token[0][1])
        # if text.shape[0] > 10:
        #     text = text[:10, :]

        # clip
        with torch.no_grad():
            text_token = clip.tokenize(text, context_length=self.token_len, truncate=True).squeeze()
        text_mask = text_token != 0 
        text_mask = text_mask.int()
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # sample = (image, mask, text_token, text_mask)
        sample = {'image': image, 'label': mask, 'text_token': text_token, 'text_mask': text_mask, "text": text}
        
        if self.mode=="train":
            sample = self.BioMedicalGaussianBlur.transform(sample)
            sample = self.PhotoMetricDistortion.transform(sample)

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
