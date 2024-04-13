# coding=utf-8
"""
    @project: smallcap-main
    @Author：no-zjc
    @file： Net_Utils.py
    @date：2024/1/22 6:24
"""
import torch
from torch import nn
import numpy as np
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
# from transformers import AutoTokenizer


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class Generation_Tool():

    @staticmethod
    def get_max_no_hallucination(result_list, k=2):
        '''
        找打验证结果中幻觉概率最小的前k个结果的下标，并返回下标和数组的值
        '''

        # 初始化保存结果的列表
        top_k_list = []
        result_list_1 = np.array(result_list)

        # 使用快速选择算法找到第一个元素前 k 大的子数组的下标
        first_element_values = result_list_1[:, 0]  # 获取所有子数组的第一个元素
        top_k_indices = np.argpartition(-first_element_values, k-1)[:k]  # 使用快速选择算法找到前 k 大的元素的索引

        # 打印找到的前 k 个元素最大的子数组的下标
        for idx in top_k_indices:
            top_k_list.append(result_list_1[idx])

        return top_k_indices, top_k_list

    @staticmethod
    def get_word_type(word):
        # 进行词性标注
        tokens = word_tokenize(word)
        tags = pos_tag(tokens)

        # 判断词语是名词还是动词
        for tag in tags:
            if tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                return 1

            elif tag[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                return 2

            else:
                return 0

    @staticmethod
    def get_sg_candidate_words(visual_constraints, cur_word_label, top_k=100):
        if cur_word_label == 0:
            return None

        if cur_word_label == 1:
            obj_info_list = visual_constraints["object"]
            obj_list = []
            for item in obj_info_list:
                obj_list.append(item["object"])
            if len(obj_list) > top_k:
                obj_list = obj_list[:top_k]
            return obj_list

        if cur_word_label == 2:
            rel_info_list = visual_constraints["relation"]
            rel_list = []
            for item in rel_info_list:
                rel_list.append(item["relation"])
            rel_list = list(set(rel_list))
            if len(rel_list) > top_k:
                rel_list = rel_list[:top_k]
            return rel_list

    @staticmethod
    def get_cur_word_token_length(tokenizer, infer_token, cur_word):
        for i in range(len(infer_token)):
            last_word_token = infer_token[len(infer_token) - i - 1:]
            last_word = tokenizer.batch_decode([last_word_token])
            if last_word[0].replace(' ', '') == cur_word:
                cur_word_token_length = i + 1
                return cur_word_token_length


# if __name__ == '__main__':
#     infer_token = [64, 3290, 318, 4273, 889]
#     sen_token = [[64, 3290, 318, 4273, 889, 257, 3290, 13]]
#
#     all_token = [[18925, 5861, 905, 198, 198, 64, 3797, 27714, 510, 1028,
#                   262, 4676, 6506, 7405, 198, 198, 64, 7586, 39145, 3797,
#                   27714, 663, 1767, 1028, 257, 6506, 1232, 198, 198, 272,
#                   29012, 1310, 3290, 290, 3797, 651, 4273, 1513, 416, 511,
#                   4870, 198, 198, 64, 582, 4273, 889, 257, 1402, 3290,
#                   290, 16755, 329, 257, 4676, 13, 198, 198, 1212, 2008,
#                   2523, 220, 64, 3290, 318, 4273, 889, 257, 3290, 13]]
#
#     tokenizer_gpt2 = AutoTokenizer.from_pretrained("../../loc_gpt2")
#
#     # infet_preds = tokenizer_gpt2.batch_decode(infer_token)
#     # sen_preds = tokenizer_gpt2.batch_decode(sen_token)
#     # all_preds = tokenizer_gpt2.batch_decode(all_token)
#     i = Generation_Tool.get_cur_word_token_length(tokenizer_gpt2, infer_token, 'petting')
#     input()