import os
import random
from typing import List, Union, Any

import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
from utils import get_imgs
import pandas as pd
import pickle


class PrototypicalCUB(Dataset):
    """
        PrototypicalBatchSampler: yield a batch of indexes at each iteration.
        Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
        In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
        for 'classes_per_it' random classes.
        __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, path, mode='train', transform=None, target_transform=None):
        self.root = path
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.images_path = {}
        self.class_ids = {}
        self.data_id = []

        # storing the path of each image with image id as the key
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[int(image_id)] = path

        # storing the class of each image with image id as the key
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[int(image_id)] = int(class_id)

        # checking for train or test modes and accordingly populating data_id array with train/test image_ids
        if self.mode == 'test':
            with open(os.path.join(self.root, 'train_val_test_split.txt')) as f:
                for line in f:
                    image_id, flag = line.split()
                    if flag == '0':
                        self.data_id.append(int(image_id))

        if self.mode == 'train':
            with open(os.path.join(self.root, 'train_val_test_split.txt')) as f:
                for line in f:
                    image_id, flag = line.split()
                    if flag == '1':
                        self.data_id.append(int(image_id))

        if self.mode == 'val':
            with open(os.path.join(self.root, 'train_val_test_split.txt')) as f:
                for line in f:
                    image_id, flag = line.split()
                    if flag == '2':
                        self.data_id.append(int(image_id))

    def __len__(self):
        return len(self.data_id)

    def get_class_by_id(self, image_id):
        return self.class_ids[image_id]

    def get_path_by_id(self, image_id):
        return self.images_path[image_id]

    def __getitem__(self, image_id):
        """
        :param image_id: image_ids from sampler
        :return: image and its corresponding labels
        """
        class_id = int(self.get_class_by_id(image_id.item())) - 1
        path = self.get_path_by_id(image_id.item())
        image = Image.open(os.path.join(self.root, 'images', path)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            class_id = self.target_transform(class_id)

        return image, class_id


class CUBSiamese(Dataset):

    def __init__(self, path, is_train, test_classes, transform=None):
        self.root = path
        self.is_train = is_train
        self.test_classes = test_classes
        self.transform = transform
        self.images_path = {}
        self.class_ids = {}
        self.data_id = []

        # storing the path of each image with image id as the key
        with open(os.path.join(self.root, 'images.txt')) as f:
            for line in f:
                image_id, path = line.split()
                self.images_path[image_id] = path

        # storing the class of each image with image id as the key
        with open(os.path.join(self.root, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = line.split()
                self.class_ids[image_id] = class_id

        # checking if data is of train or test class
        if self.is_train:
            for image_id, path in self.images_path.items():
                if self.class_ids[image_id] not in self.test_classes:
                    self.data_id.append(image_id)

        if not self.is_train:
            for image_id, path in self.images_path.items():
                if self.class_ids[image_id] in self.test_classes:
                    self.data_id.append(image_id)

    def __len__(self):
        return len(self.data_id)

    def __getitem__(self, index):

        # get first image of the image-pair along with its class
        img0_id = self.data_id[index]
        img0_class = self.class_ids[img0_id]
        img0_tuple = (img0_id, img0_class)

        # 50% of image pair should be same class and 50% different
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            while True:
                # Look until the same class image is found
                img1_id = random.choice(self.data_id)
                class_id = self.class_ids[img1_id]
                if img0_tuple[1] == class_id:
                    img1_tuple = (img1_id, class_id)
                    break
        else:
            while True:
                # Look until a different class image is found
                img1_id = random.choice(self.data_id)
                class_id = self.class_ids[img1_id]
                if img0_tuple[1] != class_id:
                    img1_tuple = (img1_id, class_id)
                    break

        # get image pair path
        image0_path = self.images_path[img0_tuple[0]]
        image1_path = self.images_path[img1_tuple[0]]

        # get the image ndarrays
        image0 = Image.open(os.path.join(self.root, 'images', image0_path)).convert('RGB')
        image1 = Image.open(os.path.join(self.root, 'images', image1_path)).convert('RGB')

        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)

        return image0, image1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))


class MultimodalCUB(Dataset):

    def __init__(self, data_dir, branch_num, split='train', embedding_type='cnn-rnn',
                 base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.imsize = []
        self.branch_num = branch_num
        for i in range(branch_num):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = self.load_bbox()
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.captions = self.load_all_captions()

        if split == 'train':
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        def load_captions(caption_name):  # self,
            cap_path = caption_name
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
            captions = [cap.replace("\ufffd\ufffd", " ")
                        for cap in captions if len(cap) > 0]
            return captions

        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, encoding='latin1')
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        wrong_ix = random.randint(0, len(self.filenames) - 1)
        if self.class_id[index] == self.class_id[wrong_ix]:
            wrong_ix = random.randint(0, len(self.filenames) - 1)
        wrong_key = self.filenames[wrong_ix]
        if self.bbox is not None:
            wrong_bbox = self.bbox[wrong_key]
        else:
            wrong_bbox = None
        wrong_img_name = '%s/images/%s.jpg' % \
                         (data_dir, wrong_key)
        wrong_imgs = get_imgs(wrong_img_name, self.imsize,
                              wrong_bbox, self.transform, normalize=self.norm)

        embedding_ix = random.randint(0, embeddings.shape[0] - 1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)

        return imgs, wrong_imgs, embedding, key  # captions

    def prepair_test_pairs(self, index):
        key = self.filenames[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_path=img_name, imsize=self.imsize, branch_num= self.branch_num,
                        bbox=bbox, transform= self.transform, normalize=self.norm)

        if self.target_transform is not None:
            embeddings = self.target_transform(embeddings)

        return imgs, embeddings, key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)
