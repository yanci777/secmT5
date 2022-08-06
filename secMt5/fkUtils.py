import torch.nn.functional as F
import torch
import torch.nn as nn
import random
import numpy
import datetime
import warnings
import logging
import os.path
import sys
import multiprocessing

import numpy as np


def attention(a, x):
    # 结果是 对a感知的 x表示 最后结果和x同shape
    # 内积计算注意力分数
    # print(a.size())
    # print(x.size())
    # print(torch.mean(a, dim=0).size())

    scores = x.bmm(a.transpose(1, 2))
    alpha = nn.functional.softmax(scores, dim=-1)
    attend = alpha.bmm(a)
    return attend


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # 返回 hh:mm:ss 形式的时间
    return str(datetime.timedelta(seconds=elapsed_rounded))


def remove_punctuation(in_str):
    in_str = str(in_str).encode('utf-8').decode('utf-8').lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』', '(', ')', '【', '●', '△', ' ', '\n', '??????']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def pre_knw_list():
    list = []
    data_path = './dataset/fk_diy_dict.txt'
    with open(data_path, "r", encoding='utf-8') as f:  # 打开文件
        data = f.readlines()  # 读取文件
    for word in data:
        word = word.replace("\n", "")
        list.append(word)
    return list


def copy_generate(cq_ids, summary_ids):
    # 如果summary_ids中含有special token 则随即替换为input_ids </s>不替换(2)
    special_token_ids = [0, 1, 3, 4]
    cq_len = cq_ids.squeeze(0).shape[0]
    # 默认<s>标签开始
    cp_summary_ids = [0]
    for id in summary_ids.squeeze(0)[1:]:
        if id in special_token_ids:
            ran_index = random.randint(1, cq_len - 3)
            cp_summary_ids.append(cq_ids[:, 1:-1].squeeze(0)[ran_index].item())
        else:
            cp_summary_ids.append(id.item())
    return cp_summary_ids


def copy_single_generate(cq_ids, pre_ids):
    unk_token_ids = 1
    cq_len = cq_ids.squeeze(0).shape[0]
    cp_summary_ids = []
    # 如果preinput_ids中含有 special token 则随即替换为cq_ids
    if pre_ids.squeeze(0)[0].item() in range(1, 7):
        ran_index = random.randint(1, cq_len - 3)
        cp_summary_ids.append(cq_ids[:, 1:-1].squeeze(0)[ran_index].item())
        print(' specail token == ', pre_ids, '  被替换为 ', cp_summary_ids)
    else:
        cp_summary_ids.append(pre_ids.item())
    return torch.LongTensor(cp_summary_ids).unsqueeze(0)


def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
    """
    top_k或top_p解码策略，仅保留top_k个或累积概率到达top_p的标记，其他标记设为filter_value，后续在选取标记的过程中会取不到值设为无穷小。
    Args:
        logits: 预测结果，即预测成为词典中每个词的分数
        top_k: 只保留概率最高的top_k个标记
        top_p: 只保留概率累积达到top_p的标记
        filter_value: 过滤标记值

    Returns:

    """
    # logits的维度必须为2，即size:[batch_size, vocab_size]
    assert logits.dim() == 2
    # 获取top_k和字典大小中较小的一个，也就是说，如果top_k大于字典大小，则取字典大小个标记
    top_k = min(top_k, logits[0].size(-1))
    # 如果top_k不为0，则将在logits中保留top_k个标记
    if top_k > 0:
        # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的top_k标记
        for i in range(logits.size(0)):  # logits.size(0)来获取logits的元素个数
            logit = logits[i]
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value
    # 如果top_p不为0，则将在logits中保留概率值累积达到top_p的标记
    if top_p > 0.0:
        # 对logits进行递减排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # 对排序后的结果使用softmax归一化，再获取累积概率序列
        # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 删除累积概率高于top_p的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动，使第一个标记也保持在top_p之上
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            temp = logits[index]
            # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的累积概率达到top_p的标记
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            temp[indices_to_remove] = filter_value
    return logits


import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def weighted_sum(word_vec, word_dim, device):
    b = nn.Linear(word_dim, 1).to(device)  # 参数张量
    # 内积得分，维度是batch x seq_len x 1
    scores = b(word_vec).to(device)
    # softmax运算，结果维度是batch x seq_len x 1
    weights = F.softmax(scores, dim=1).to(device)
    # 用矩阵乘法实现加权和，结果维度是batch x word_dim x 1
    res = torch.bmm(word_vec.transpose(1, 2), weights).to(device)
    # 删除最后一维，结果维度是batch x word_dim
    res = res.squeeze(2)
    return res


def add_pre_knw(word_id, word_vec, word_dim, device):
    pre_knw_list = 7993
    # 塔利班 伊斯兰国 基地组织 枪击 炸弹 袭击

    b = nn.Linear(word_dim, 1).to(device)  # 参数张量
    # 内积得分，维度是batch x seq_len x 1
    scores = b(word_vec).to(device)
    # softmax运算，结果维度是batch x seq_len x 1
    weights = F.softmax(scores, dim=1).to(device)
    print(word_id)

    for index in range(0, word_id.shape[1]):
        if word_id[:, index].item() in range(pre_knw_list, 7999):
            print('存在 重点token id == ', word_id[:, index].item())
            word_vec[:, index] = word_vec[:, index] * (1 + weights[:, index])


def build_sentence_vector(sentence, size, w2v_model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in sentence:
        try:
            vec += w2v_model.wv[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# 计算两个句向量的余弦相似性值
def cosine_similarity(vec1, vec2):
    vector_a = np.mat(vec1)
    vector_b = np.mat(vec2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom
    return sim


# 输入两个句子，计算两个句子的余弦相似性
def compute_cosine_similarity(sents_1, sents_2):
    size = 50
    # w2v_model=Word2Vec.load('w2v_model.pkl')
    # w2v_model = Word2Vec.load('corpusSegDone_1.model')
    w2v_model = Word2Vec.load('./fk_t5_eval/corpusSegDone_1.model')
    vec1 = build_sentence_vector(sents_1, size, w2v_model)
    vec2 = build_sentence_vector(sents_2, size, w2v_model)
    similarity = cosine_similarity(vec1, vec2)
    return similarity


def check_ques_type(ques_token_list, answer_token_list):
    # 若问题为是非类型 判断答案是否包含是非类型关键词
    # 若包含则不处理，否则计算答案和是非类型关键词的语义相似度，选取相似度最大的替换为答案
    qa_type = {
        "ques_isTF": False,
        "ques_keyword": None,
        "ans_has_keyword": False
    }
    ques_keyword_list = ["是否", "是不是", "有没有"]
    ans_keyword_list = ["是", "否", "有", "没有", "尚未", "已", "已经"]
    for token in ques_token_list:
        if token in ques_keyword_list:
            qa_type["ques_isTF"] = True
            qa_type["ques_keyword"] = token

            for item in answer_token_list:
                if item in ans_keyword_list:
                    print("--- 答案包含是非关键词 ---")
                    qa_type["ans_has_keyword"] = True

    return qa_type


def change_tf_answer(model_answer, ques_keyword):
    similarity_list = []
    ques_keyword_list = [["是否", "是不是"], ["有没有"], ["已经"]]
    ans_keyword_list = [["是", "否"], ["有", "没有"], ["尚未", "已", "已经"]]

    ques_type = 0
    for index in range(0, 3):
        if ques_keyword in ques_keyword_list[index]:
            ques_type = index

    for word in ans_keyword_list[ques_type]:
        similarity_list.append(compute_cosine_similarity(model_answer, word))
    new_answer_index = similarity_list.index(min(similarity_list))
    new_answer = ans_keyword_list[ques_type][new_answer_index]

    return new_answer
    # print("新答案为  ===  ", new_answer)


# 处理无法回答问题向量
def trans_ques_vec(ques_cls, word_dim, device):
    b = nn.Linear(word_dim, 1).to(device)  # 参数张量
    # 内积得分，维度是batch x seq_len x 1
    scores = b(ques_cls).to(device)
    # softmax运算，结果维度是batch x seq_len x 1
    weights = F.softmax(scores, dim=1).to(device)