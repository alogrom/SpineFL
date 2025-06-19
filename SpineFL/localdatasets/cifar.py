import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import check_exist, save, load
class CIFAR10(Dataset):
    def __init__(self, cifar_path, split, subset, transform=None):
        self.path = cifar_path
        self.subset = subset
        self.split = split
        self.transform = transform
        if not check_exist(self.processed_folder):
            self.process()
        self.img, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[subset]
        self.classes = load(os.path.join(self.processed_folder, 'meta.pt'))

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform is not None:
            img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
            img = self.transform(img)
        return {'img': img, self.subset: target}

    def __len__(self):
        return len(self.target)

    @property
    def processed_folder(self):
        return os.path.join(self.path, 'processed')

    def process(self):
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def make_data(self):
        train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_filenames = ['test_batch']
        train_img, train_label = read_pickle_file(self.path, train_filenames)
        test_img, test_label = read_pickle_file(self.path, test_filenames)
        train_target, test_target = {'label': train_label}, {'label': test_label}
        with open(os.path.join(self.path, 'batches.meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        classes = {}
        for i, item in enumerate(data['label_names']):
            classes[i] = item
        return (train_img, train_target), (test_img, test_target), classes


class CIFAR100(CIFAR10):

    def make_data(self):
        train_filenames = ['train']
        test_filenames = ['test']
        train_img, train_label = read_pickle_file(self.path, train_filenames)
        test_img, test_label = read_pickle_file(self.path, test_filenames)
        train_target, test_target = {'label': train_label}, {'label': test_label}
        with open(os.path.join(self.path,  'meta'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        classes = {}
        for i, item in enumerate(data['fine_label_names']):
            classes[i] = item
        return (train_img, train_target), (test_img, test_target), classes


def read_pickle_file(path, filenames):
    img, label = [], []
    for filename in filenames:
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            img.append(entry['data'])
            label.extend(entry['labels']) if 'labels' in entry else label.extend(entry['fine_labels'])
    img = np.vstack(img).reshape(-1, 3, 32, 32)
    return img, label




