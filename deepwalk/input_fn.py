# -*- coding: utf-8 -*-
import random
import torch

import numpy as np 
import torch.utils.data as Data

from utils import *

class InputData(object):
    def __init__(self, window_size, batch_size, walks):
        self.vocab_size = None
        self.word_frequency = None
        self.batch_size = batch_size
        self.window_size = window_size
        self.walks = walks
        self.sample_table = self.init_sample_table()
        self.train_set = self.construct_train_set(self.walks, window_size)
        self.train_set = Data.TensorDataset(torch.Tensor(self.train_set))
        self.J, self.q = alias_setup(self.sample_table)
        self.data_iter = Data.DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=10)
        
    def init_sample_table(self):
        word_frequency = dict()
        for line in self.walks:
            for w in line:
                if w in word_frequency:
                    word_frequency[w] += 1
                else:
                    word_frequency[w] = 1
        self.word_frequency = word_frequency
        self.vocab_size = len(word_frequency)

        # sort word by index
        pow_frequency = np.array([v[1]**0.75 for v in sorted(self.word_frequency.items(), key=lambda x: x[0])])
        words_pow = sum(pow_frequency)
        pow_frequency = pow_frequency / words_pow
        return pow_frequency
    
     # 构造训练集
    def construct_train_set(self, data, max_window):
        print("Construct train set")
        res = []
        for session in data:
            centers, contexts = [], []
            centers += session
            for center_i in range(len(session)):
                window_size = random.randint(1, max_window)
                indices = list(range(max(0, center_i - window_size),
                    min(len(session), center_i + 1 + window_size)))
                indices.remove(center_i)
                contexts.append([session[idx] for idx in indices])
            assert len(centers) == len(contexts)
            for i, v in enumerate(contexts):
                res += zip([centers[i]] * len(v), v)
        print(" ~done.")
        return res

    # 负采样
    def get_negative_sample(self, pos_pair, negative_num):
        neg_pair = np.zeros((negative_num))
        for pair in pos_pair:
            neg_v = []
            while len(neg_v) < negative_num:
                neg_word = alias_draw(self.J, self.q)
                if neg_word != pair[0]:
                    neg_v.append(neg_word)
                else:
                    continue
            neg_pair = np.vstack([neg_pair, neg_v])
        return neg_pair[1:]   
