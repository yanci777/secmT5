from transformers import MT5ForConditionalGeneration, T5Tokenizer, MT5Model
from t5_tokenizer import T5PegasusTokenizer
import fkUtils
import torch


class Fk_MT5Model(MT5ForConditionalGeneration):
    device = None

    def __init__(self, config):
        super().__init__(config)
        self.transformer = MT5ForConditionalGeneration(config)

    def get_device(self, device):
        self.device = device

    def forward(self, ques_ids=None, text_ids=None, ans_ids=None):
        ques_embeds = self.transformer.encoder.embed_tokens(ques_ids).to(self.device)
        text_embeds = self.transformer.encoder.embed_tokens(text_ids).to(self.device)
        # 计算问题对文本的注意力 文本被注意 结果为 batch * ques_len * dim
        context_att = fkUtils.attention(text_embeds, ques_embeds)
        # 将注意力计算结果和文本embed拼接成输入格式
        ques_text_embeds = torch.cat([context_att, text_embeds[:, 1:, :]], dim=1)

        transformer_outputs = self.transformer(inputs_embeds=ques_text_embeds,
                                               labels=ans_ids)
        return transformer_outputs
