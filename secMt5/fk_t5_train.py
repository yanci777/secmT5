import torch

import os
import time

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup

from transformers import WEIGHTS_NAME, CONFIG_NAME
import dataProcessor
import extra_dataProcessor
import cmrc_dataProcessor
import torch.nn.functional as F

import torch.nn as nn
from t5_tokenizer import T5PegasusTokenizer
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers import (
    DataProcessor,
    BertTokenizer,
    squad_convert_examples_to_features,
    BertForQuestionAnswering,
    BertForPreTraining,
    BartConfig,
    BartTokenizer,
    BartForConditionalGeneration, BartConfig
)
import fkUtils

epochs = 4


def train_training(train_data, tokenizer, epoch_i, model):
    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()
    # net.train()

    for step, batch in enumerate(train_data):
        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:
            elapsed = fkUtils.format_time(time.time() - t0)
            print('epochs == ', epoch_i + 1,
                  ' ---  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_data), elapsed))

        question = fkUtils.remove_punctuation(batch.question_text)
        context = fkUtils.remove_punctuation(batch.context_text)
        answer = fkUtils.remove_punctuation(batch.answer_text)

        ques_inputs = tokenizer.encode_plus(question,
                                            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                            max_length=512,
                                            return_tensors='pt',
                                            truncation=True)
        ques_ids = ques_inputs['input_ids'].to(device)

        text_inputs = tokenizer.encode_plus(context,
                                            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                            max_length=512,
                                            return_tensors='pt',
                                            truncation=True)
        text_ids = text_inputs['input_ids'].to(device)

        ans_inputs = tokenizer.encode_plus(answer,
                                           add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                           max_length=512,
                                           return_tensors='pt',
                                           truncation=True)
        ans_ids = ans_inputs['input_ids'].to(device)

        # 清空梯度
        model.zero_grad()
        """
            拼接embedding输入
        """
        ques_embeds = model.encoder.embed_tokens(ques_ids).to(device)
        text_embeds = model.encoder.embed_tokens(text_ids).to(device)
        # 计算问题对文本的注意力 文本被注意 结果为 batch * ques_len * dim
        ques2context_att = fkUtils.attention(text_embeds, ques_embeds)
        # batch * context_len * dim
        context2ques_att = fkUtils.attention(ques_embeds, text_embeds)
        # 将注意力计算结果和文本embed拼接成输入格式
        ques_text_embeds = torch.cat([ques2context_att, context2ques_att[:, 1:, :]], dim=1)

        outputs = model.forward(inputs_embeds=ques_text_embeds, labels=ans_ids)
        loss = outputs.loss
        print('loss.item() == ', loss.item())
        if loss.item() > 9 and epoch_i + 1 > 1:
            print("- - - - - - - 效果表现不好 ：", question)
            print("- - - - - - - - - - - - ：", answer)
        elif loss.item() < 0.4 and epoch_i + 1 > 1:
            print("- - - - - - - 效果表现好 ：", question)
            print("- - - - - - - - - - -  ：", answer)

        total_train_loss += loss.item()

        # backward 更新 gradients.
        loss.backward()

        # 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新模型参数
        optimizer.step()

        # 更新 learning rate.
        scheduler.step()

    # 计算batches的平均损失.
    avg_train_loss = total_train_loss / len(train_data)

    print("  平均训练损失 loss: {0:.2f}".format(avg_train_loss))
    return avg_train_loss


def train_evalution(dev_data, tokenizer, model):
    total_eval_loss = 0
    model.eval()

    for step, batch in enumerate(dev_data):
        question = fkUtils.remove_punctuation(batch.question_text)
        context = fkUtils.remove_punctuation(batch.context_text)
        answer = fkUtils.remove_punctuation(batch.answer_text)

        ques_inputs = tokenizer.encode_plus(question,
                                            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                            max_length=512,
                                            return_tensors='pt',
                                            truncation=True)
        ques_ids = ques_inputs['input_ids'].to(device)

        text_inputs = tokenizer.encode_plus(context,
                                            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                            max_length=512,
                                            return_tensors='pt',
                                            truncation=True)
        text_ids = text_inputs['input_ids'].to(device)

        ans_inputs = tokenizer.encode_plus(answer,
                                           add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                           max_length=512,
                                           return_tensors='pt',
                                           truncation=True)
        ans_ids = ans_inputs['input_ids'].to(device)

        # 在valuation 状态，不更新权值，不改变计算图
        with torch.no_grad():
            ques_embeds = model.encoder.embed_tokens(ques_ids).to(device)
            text_embeds = model.encoder.embed_tokens(text_ids).to(device)
            # 计算问题对文本的注意力 文本被注意 结果为 batch * ques_len * dim
            ques2context_att = fkUtils.attention(text_embeds, ques_embeds)
            context2ques_att = fkUtils.attention(ques_embeds, text_embeds)
            # 将注意力计算结果和文本embed拼接成输入格式
            ques_text_embeds = torch.cat([ques2context_att, context2ques_att[:, 1:, :]], dim=1)

            outputs = model.forward(inputs_embeds=ques_text_embeds, labels=ans_ids)
            loss = outputs.loss

            print('loss.item() == ', loss.item())
            # 计算 validation loss.
            total_eval_loss += loss.item()

    return total_eval_loss, len(dev_data)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('there are %d GPU(s) available.' % torch.cuda.device_count())
        print('we will use the GPU: ', torch.cuda.get_device_name(0))
    else:
        print('No GPU availabel, using the CPU instead.')
        device = torch.device('cpu')

    data_dir = 'dataset'

    t5_model_path = 'imxly/t5-pegasus'
    local_t5_path = 'fk_mt5_pegasus'
    model_path = local_t5_path

    processor = dataProcessor.MySquadProcessor()
    Train_data = processor.get_train_examples(data_dir)
    Dev_data = processor.get_dev_examples(data_dir)

    tokenizer = T5PegasusTokenizer.from_pretrained(model_path,
                                                   vocab_file='dataset/vocab_pro.txt'
                                                   )
    pre_kn_list = fkUtils.pre_knw_list()
    tokenizer.add_special_tokens({'additional_special_tokens': pre_kn_list})

    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)

    # training steps 的数量: [number of batches] x [number of epochs].
    total_steps = len(Train_data) * epochs

    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # args.learning_rate - 默认是 5e-5
                      eps=1e-8  # args.adam_epsilon  - 默认是 1e-8， 是为了防止衰减率分母除到0
                      )

    warm_up_ratio = 0.1  # 定义要预热的step
    # 设计 learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warm_up_ratio * total_steps,
                                                num_training_steps=total_steps)  # Default value in run_glue.py

    # 设置总时间
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

        # ========================================
        #               training
        # ========================================
        t0 = time.time()
        avg_train_loss = train_training(Train_data, tokenizer, epoch_i, model)
        # 计算训练时间.
        training_time = fkUtils.format_time(time.time() - t0)
        print("  训练时间: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        t0 = time.time()

        total_eval_loss, valid_dataloader_length = train_evalution(Dev_data, tokenizer, model)

        print("")

        # 计算batches的平均损失.
        avg_val_loss = total_eval_loss / valid_dataloader_length

        # 计算validation 时间.
        validation_time = fkUtils.format_time(time.time() - t0)

        print("  平均测试损失 Loss: {0:.2f}".format(avg_val_loss))
        print("  测试时间: {:}".format(validation_time))

    print("训练一共用了 {:} (h:mm:ss)".format(fkUtils.format_time(time.time() - total_t0)))

    output_dir = "./fk_gmodel/"
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model.state_dict(), output_model_file)
    model.config.to_json_file(output_config_file)

    # state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epochs}
    # torch.save(net, output_dir + 'fknetpara.pth')
