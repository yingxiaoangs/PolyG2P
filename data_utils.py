import json5
import numpy as np
import torch
import os

import symbols

# 统计词频 反向计算权重 目的是解决低频词的预测问题
class PolyCounter:
    def __init__(self):
        self.poly_counter = 'data/poly_counter.json5'
        poly_file = open('data/poly_phone.json5', 'r')
        self.poly_dist = json5.load(poly_file)
        self.counts = {}
        self.num_classes = len(symbols.pinyin2id)

    def process_char(self, char, target):
        if char not in self.counts:
            candidates = self.poly_dist.get(char)
            self.counts[char] = {i: 0 for i in candidates}
            self.counts[char][target] += 1
        else:
            self.counts[char][target] += 1

        with open(self.poly_counter, 'w') as f:
            json5.dump(self.counts, f)

    def class_weights(self, char):
        if os.path.exists(self.poly_counter):
            with open(self.poly_counter) as f:
                self.counts = json5.load(f)
        char_counts = self.counts.get(char)
        valid_indices = [symbols.pinyin2id.get(i) for i in char_counts.keys()]
        valid_counts = [k for k in char_counts.values()]

        epsilon = 1e-6  # 防止除零
        inv_counts = 1.0 / (np.array(valid_counts) + epsilon)

        # 归一化权重
        sum_inv = inv_counts.sum()
        normalized_weights = inv_counts / sum_inv

        class_weights = torch.zeros(self.num_classes)  # 初始化为全零

        # 填充有效类别的权重
        for idx, weight in zip(valid_indices, normalized_weights):
            class_weights[idx] = weight
            
        return class_weights



