import os
import warnings
import logging
import os.path
import sys
import multiprocessing

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

path = "/etc/yanci/MyDrive/t5_model/"
os.chdir(path)


# word2vec_model.most_similar('袭击')

# 对每个句子的所有词向量取均值，来生成一个句子的vector
# sentence是输入的句子，size是词向量维度，w2v_model是训练好的词向量模型
def build_sentence_vector(sentence, size, w2v_model):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in sentence:
        try:
            vec += w2v_model[word].reshape((1, size))
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
    w2v_model = Word2Vec.load('corpusSegDone_1.model')
    vec1 = build_sentence_vector(sents_1, size, w2v_model)
    vec2 = build_sentence_vector(sents_2, size, w2v_model)
    similarity = cosine_similarity(vec1, vec2)
    return similarity


real_list = [
    "枪击事件",
    "造成至少２０人死亡",
    "香榭丽莎大道遭遇恐怖袭击",
    "埃尔比勒国际机场遭遇无人机袭击",
    "恐怖袭击事件",
    "索马里青年党",
    "索马里青年党",
    "炸弹袭击",
    "伊斯兰国",
    "伊斯兰国",
    "还没有",
    "警方还没有掌握枪手的详细信息",
    "是",
    "还不明确",
    "是",
]

model_list = [
    "枪击事件",
    "地雷爆炸伤亡如何",
    "香榭丽舍大道",
    "携带炸药的无人机袭击",
    "是一起恐怖袭击行为",
    "索马里目前",
    "索马里反对派",
    "发生爆炸事件已经造成两人死亡约10人受伤",
    "俄罗斯联邦安全局挫败了一个恐怖组织的阴谋",
    "11人",
    "目前还没有组织或个人宣称制造",
    "警方尚未",
    "3日连续遭遇两起汽车炸弹袭击",
    "他是法籍阿尔及利亚人",
    "因为一名女性身缠炸药自爆",
]
for index in range(len(real_list)):
    print("** " * 50)
    print("句子相似度为 == ", compute_cosine_similarity(real_list[index], model_list[index]))