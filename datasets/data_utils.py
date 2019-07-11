"""
Created on Jul 11, 2019

Module data loading

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import torch
from PIL import Image


# sim_pairs - random pairs of images from same person, labeled 1
# diff_pairs - random pairs of images from different people
# apn (anchor-pos-neg)-triplets of anchor, pos and negative, no label
# start, finish select range of folders to train/test

def get_pairs(path, start, finish):
    sim_pairs = []
    diff_pairs = []
    apn = []
    root_path = path

    folder_list = [i for i in range(start, finish)]

    for i in range(start, finish):

        folder_list_new = folder_list.copy()

        # Not to repeat the person in diff_pairs, remove ith folder in the new_list
        folder_list_new.remove(i)

        for j in range(1, 11):
            # random.sample selects rand index --> selects rand img
            idx = random.sample(range(1, 11), 2)
            diff_person = random.sample(folder_list_new, 1)

            sim_pairs.append(['s' + str(i) + '\\' + str(idx[0]) + '.pgm',
                              's' + str(i) + '\\' + str(idx[1]) + '.pgm', 1])

            diff_pairs.append(['s' + str(i) + '\\' + str(idx[0]) + '.pgm',
                               's' + str(diff_person[0]) + '\\' + str(idx[1]) + '.pgm', 0])

            apn.append(['s' + str(i) + '\\' + str(idx[0]) + '.pgm',
                        's' + str(i) + '\\' + str(idx[1]) + '.pgm',
                        's' + str(diff_person[0]) + '\\' + str(idx[1]) + '.pgm'])

    return sim_pairs, diff_pairs, apn


# FOR TEST ONLY
# For test we want images, tensors, embeds in order so that we know which one is which

def get_test_pairs(start, finish):
    test_pairs = []

    folder_list = [i for i in range(start, finish)]

    for i in range(start, finish):

        folder_list_new = folder_list.copy()

        for j in range(1, 11):
            test_pairs.append(['s' + str(i) + '\\' + str(j) + '.pgm'] * 3)

    return test_pairs


# given list of pairs or triplets and batch size it finds total number of batches
def get_num_batches(the_list, batch_size):
    list_length = len(the_list)

    if list_length % batch_size == 0:
        num_batches = list_length // batch_size
    else:
        num_batches = list_length // batch_size + 1
    return num_batches


# given list of pairs, batch size, num_batches it retrieves all the pairs that belongs to i_th batch
def get_batch(i, batch_size, num_batches, the_list):
    if i < num_batches - 1:

        cur_batch = the_list[i * batch_size:i * batch_size + batch_size]

    else:
        cur_batch = the_list[i * batch_size:]

    return cur_batch


# Remember to rectify root path
def get_input_tensors(cur_batch):
    from torchvision.transforms import ToTensor

    # 3 tensors are initialized and to each respective image tensors are added
    batch_imgs_anc = torch.Tensor()
    batch_imgs_pos = torch.Tensor()
    batch_imgs_neg = torch.Tensor()

    # root_path is simply path to folderls s1 to s40
    root_path = 'C:\\Users\\...\\atnt_faces\\'

    # cur_batch is subset of triplets with length of batch_size
    # cur_batch has 3 items in each sublist -->[0]: anchor, [1]:positive, [2]:negative samples
    for i in range(len(cur_batch)):
        file_path_anc = cur_batch[i][0]
        file_path_pos = cur_batch[i][1]
        file_path_neg = cur_batch[i][2]
        new_tensor_anc = ToTensor()(Image.open(root_path + file_path_anc)).unsqueeze(0)
        new_tensor_pos = ToTensor()(Image.open(root_path + file_path_pos)).unsqueeze(0)
        new_tensor_neg = ToTensor()(Image.open(root_path + file_path_neg)).unsqueeze(0)
        batch_imgs_anc = torch.cat((batch_imgs_anc, new_tensor_anc), dim=0)
        batch_imgs_pos = torch.cat((batch_imgs_pos, new_tensor_pos), dim=0)
        batch_imgs_neg = torch.cat((batch_imgs_neg, new_tensor_neg), dim=0)

    return batch_imgs_anc, batch_imgs_pos, batch_imgs_neg
