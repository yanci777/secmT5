# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: model.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:26
"""
    文件说明：
    GPT2模型文件，主要对transformers包中GPT2LMHeadModel的重写，修改计算loss部分，只计算预测title部分的loss
"""
import copy
import math
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from t5_tokenizer import T5PegasusTokenizer
import fkUtils
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('there are %d GPU(s) available.' % torch.cuda.device_count())
    print('we will use the GPU: ', torch.cuda.get_device_name(0))
else:
    print('No GPU availabel, using the CPU instead.')
    device = torch.device('cpu')


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
        print("注意力shape == ", context_att.shape)

        # 将注意力计算结果和文本embed拼接成输入格式
        ques_text_embeds = torch.cat([context_att, text_embeds[:, 1:, :]], dim=1)
        print(ques_text_embeds.shape)
        transformer_outputs = self.transformer(inputs_embeds=ques_text_embeds,
                                               labels=ans_ids)
        print(transformer_outputs.loss)


if __name__ == '__main__':
    sample_list = [
        {
            'context': "当地时间6月16日，美国马里兰州巴尔的摩市发生枪击事件，造成1人死亡，5人受伤。巴尔的摩警察局长哈里森（Michael Harrison）表示，枪击案发生在该市街区，6名受害者在街上行走，两至三个枪手突然向受害者开枪，随后逃离现场。目前5名伤者处于危急状态。哈里森表示，此次袭击行为是“非常无耻”的，枪手突袭街区居民的行为可能造成很多人受伤。"
            , 'question': "美国马里兰州巴尔的摩市发生什么"
            , 'answer': "枪击事件"
            , 'type': "描述"
        },
        {
            'context': "这是4月9日在索马里摩加迪沙拍摄的汽车爆炸现场。位于首都摩加迪沙的索马里国防部附近9日发生自杀式汽车爆炸事件，造成至少10人死亡、多人受伤。位于首都摩加迪沙的索马里国防部附近９日发生自杀式汽车爆炸事件，造成至少１０人死亡、多人受伤。索马里国家通讯社说，这起爆炸事件造成至少１０人死亡。但一位拒绝透露姓名的当地警察通过电话告诉记者，爆炸造成至少１５人死亡，其中包括３名士兵，另有多人受伤。这是索马里近日第二次发生爆炸事件。４月６日，索马里总统穆罕默德·阿卜杜拉希·穆罕默德宣布全国进入战争状态，并敦促索马里人民与政府军一道向极端组织“青年党”宣战。６日当晚，索南部下谢贝利州发生地雷爆炸事件，造成至少２０人死亡。"
            , 'question': "地雷爆炸伤亡如何。"
            , 'answer': "造成至少２０人死亡"
            , 'type': "描述"
        },
        {
            'context': "中新网4月22日电 综合报道，法国大选即将于当地时间23日投票，在投票前3天，法国巴黎最为繁华的香榭丽莎大道遭恐袭，导致警察一死两伤，袭击者被击毙。巴黎“最美街道”上的血案给法国大选蒙上一层阴影；有分析指出，恐袭将对选民投票意向造成一定影响。当地时间20日晚9时左右，香榭丽舍大道104号附近，一名枪手向停在大道上的警车开枪并与警方交火，导致一名警察中弹身亡，两名警察受伤，另有一名路人轻伤。枪手随后被击毙。截至发稿时，除被击毙的枪手外，还有一名涉及此次恐袭的嫌犯已经在比利时主动向警方自首。极端组织“伊斯兰国”当天通过宣传机构发布消息称，对巴黎袭警枪击事件负责。"
            , 'question': "法国巴黎香榭丽莎大道发生了什么。"
            , 'answer': "香榭丽莎大道遭遇恐怖袭击"
            , 'type': "描述"
        },
        {
            'context': "海外网9月12日电 综合今日俄罗斯电视台、俄新社11日消息，“9·11”纪念日当天，美国驻伊拉克军事基地埃尔比勒国际机场遭无人机袭击，至少发生6次爆炸，基地内警报响起。社交媒体视频显示，爆炸地点火光四射，升起浓烟，美军基地内警报响起。路透社援引现场目击者的话称，机场附近听到至少6次爆炸声。俄新社则报道称，至少发生3次爆炸。伊拉克库尔德地区反恐理事会证实，“携带炸药的无人机”袭击了以美国为首的国际联军基地埃尔比勒国际机场。库尔德地区安全部门表示，火箭弹击中了机场附近。"
            , 'question': "伊拉克库尔德地区反恐理事会证实了什么。"
            , 'answer': "埃尔比勒国际机场遭遇无人机袭击"
            , 'type': "描述"
        },
        {
            'context': "最新消息：纽约曼哈顿卡车撞人恐怖袭击事件，已致8人死亡。据纽约警方消息，纽约曼哈顿下城10月31日发生卡车撞人袭击事件。纽约市长德布拉西奥说，事件已导致8人死亡、多人受伤，是一起恐怖袭击行为。纽约警方在推特上发文说，31日下午，一辆卡车在曼哈顿下城世贸中心附近冲入行人、自行车道，撞倒多人后继续行驶，并撞上另一辆车。司机随后持仿真枪下车，后被警察开枪击伤。警方说，嫌疑人目前已被拘留。今天正值西方传统节日万圣节，纽约万圣节游行的出发聚集地距离案发地只有1公里左右的距离，游行是否照常进行还不得而知。"
            , 'question': "纽约曼哈顿卡车撞人是什么事件。"
            , 'answer': "恐怖袭击事件"
            , 'type': "描述"
        },
        {
            'context': "摩加迪沙消息：索马里警方１４日表示，当晚首都摩加迪沙汽车炸弹袭击已造成１４人死亡，袭击者在爆炸发生后又挟持事发现场至少２０名人质与警方对峙。一名警方负责人说，被袭酒店餐厅入口发生爆炸后，不明数量的袭击者持枪冲入邻近另一家餐馆挟持了２０多人，并仍在与警方对峙中。截至目前，尚不知道这些人质的安全状况。警方表示，袭击者切断餐馆电力，利用夜色与警方周旋。目前，警方已经封锁该区域。摩加迪沙霍丹地区一家酒店餐厅１４日晚遭汽车炸弹袭击。据悉，袭击者驾驶一辆装有爆炸物的汽车冲进酒店，造成包括两名自杀式袭击者在内的至少１４人死亡，另有３０多人受伤。索马里“青年党”宣称制造了此次袭击。据索马里媒体报道，光顾这家酒店餐厅的顾客大多是政府官员和年轻人。"
            , 'question': "哪个组织宣称制造了此次袭击。"
            , 'answer': "索马里青年党"
            , 'type': "实体"
        },
        {
            'context': "索马里警方２３日说，索东北部邦特兰地区当天发生地雷爆炸事件，造成至少９名索士兵死亡，另有３名士兵受伤。警方说，爆炸发生在邦特兰地区巴里州加勒加拉镇，袭击目标是在该地区打击极端组织“青年党”的政府军士兵。索马里“青年党”武装声称制造了这次袭击事件。"
            , 'question': "哪个组织声称制造了这次袭击事件。"
            , 'answer': "索马里“青年党”"
            , 'type': "实体"
        },
        {
            'context': "当地时间11月13日晚，位于菲律宾首都马尼拉的菲律宾众议院大楼发生爆炸事件，已经造成两人死亡，约10人受伤。 警方调查人员说，爆炸发生在当地时间20时15分众议院休会后不久，地点位于众议院大楼南翼楼的大堂附近。爆炸产生的强烈震波摧毁了大堂的部分天花板，爆炸引发的大火还烧毁了部分停在楼外的汽车。菲警方说，爆炸的原因目前尚在调查中，但不排除恐怖袭击的可能。警方调查人员怀疑，有人在南翼楼外的花坛内安放了炸弹。 据当地媒体报道，一名在爆炸中受伤的议员在被送到医院后，因伤势过重，不治身亡，一名议员的司机在爆炸中丧生。 "
            , 'question': "菲律宾首都马尼拉的菲律宾众议院大楼遭遇了什么袭击"
            , 'answer': "炸弹袭击"
            , 'type': "实体"
        },
        {
            'context': '俄罗斯联邦安全局２６日宣布成功挫败一起恐怖袭击阴谋。这一阴谋由极端组织“伊斯兰国”支持者一手策划，目标直指俄远东地区的萨哈林岛。联邦安全局在一份声明中说，已羁押两名涉恐嫌疑人，他们密谋在萨哈林岛上的“人流密集区高调实施恐怖袭击”。萨哈林岛地理位置重要，堪称俄罗斯的“东部前哨站”。那里油气资源丰富，吸引来自俄天然气工业股份公司、壳牌石油公司等能源巨头数十亿美元投资。按照联邦安全局的说法，他们在搜查嫌疑人住所时发现了一个自制爆炸装置、多部手机以及一些“炸弹制作指南”。两名嫌疑人中，一人系“中亚某国公民”，另一人则为俄罗斯公民。'
            , 'question': "俄罗斯联邦安全局挫败了哪个恐怖组织的阴谋"
            , 'answer': "伊斯兰国"
            , 'type': "实体"
        },
        {
            'context': "巴基斯坦警方2月17日宣布，该国西南部的一个法院当天发生炸弹爆炸事件，炸弹威力巨大，造成包括一名法官在内的至少12人死亡，20多人受伤。 综合美英媒体报道，爆炸发生在巴基斯坦西南部俾路支省首府奎达市，目前尚不清楚谁是幕后黑手。当地警方官员表示，已经将死者和受伤人员运往附近医院。这名官员称，确认民事法官阿卜杜勒·瓦西德在炸弹袭击中丧生，死者中还包括6名律师。 经历这场爆炸的律师阿卜杜勒·拉希德说，他在现场看到一个头颅，因此怀疑这是一次自杀式袭击。警方则拒绝进行推测，称他们仍在展开调查。 9·11事件发生后，巴基斯坦就一直坚定地支持美国发动的反恐战争，因此也经历了一系列恐怖袭击。就在此次爆炸发生前一天，巴警方刚刚宣布在卡拉奇和拉瓦尔品第逮捕了5名嫌疑人，这些武装分子计划对外国人和什叶派穆斯林发动自杀式袭击。 "
            , 'question': "俄罗斯联邦安全局挫败了哪个恐怖组织的阴谋"
            , 'answer': "伊斯兰国"
            , 'type': "实体"
        },
        {
            'context': "伊拉克首都巴格达9日上午发生两起自杀式爆炸袭击，造成至少11人死亡、41人受伤。伊警方说，一名袭击者在巴格达贾迪达区的一家电影院附近引爆了汽车炸弹，造成至少7人死亡、30人受伤。另一起袭击发生在塔吉区的军事基地，一名袭击者在基地入口处的检查站引爆了身上炸药，造成4名士兵死亡、11人受伤。目前还没有组织或个人宣称制造这两起袭击。警方怀疑为极端组织“伊斯兰国”武装人员所为。“伊斯兰国”目前控制着伊拉克西部和北部的大片地区，并在多个省份与伊政府军激烈交战。5月23日，伊政府军展开收复西部费卢杰市的战役。“伊斯兰国”对此采取了疯狂的报复行动。"
            , 'question': "目前是否有组织或个人宣称制造这两起袭击。"
            , 'answer': "还没有"
            , 'type': "是非"
        },
        {

            'context': "据新华社休斯敦6月12日电（记者 高路）美国得克萨斯州首府奥斯汀市中心12日凌晨发生枪击事件，造成至少13人受伤，枪手目前在逃。奥斯汀警察局代局长约瑟夫·查康在新闻发布会上说，枪击事件发生在凌晨1时30分左右，地点在酒吧与餐馆林立的奥斯汀市中心。目前尚不清楚枪击原因。查康说，伤者中有两人伤势严重，所有伤者已被送往医院救治。警方尚未掌握枪手的详细信息。"
            , 'question': "警方是否掌握枪手的详细信息"
            , 'answer': "警方还没有掌握枪手的详细信息"
            , 'type': "是非"
        },
        {
            'context': "伊拉克官方表示，首都巴格达市当地时间3日连续遭遇两起汽车炸弹袭击,造成至少165人死亡、168人受伤。极端组织“伊斯兰国”(IS)宣称对第一起事件负责。据美联社报道，一名自杀式袭击者驾驶装满炸药的冷冻货车3日凌晨闯入巴格达南部卡拉达区，随后引爆炸弹。卡拉达区是巴格达的商业中心之一，随着历时一个月的斋月接近尾声，很多穆斯林在庆祝斋月结束的开斋节之前都来到该购物区消费。这起袭击事件造成的死伤者中多数为儿童。附近多家商店和多辆汽车被毁，大火在周日凌晨依然熊熊燃烧。爆炸发生后数小时，很多当地民众为发泄不满，朝前来探望的总理阿巴迪的座驾投掷石子。随后IS在社交媒体承认为卡拉达爆炸事件负责，并称这是针对什叶派信徒发动的恐怖攻击。第二起袭击则发生在巴格达东部，5人因袭击死亡、16人受伤。路透社称，医院和警方确认了上述伤亡数字，但没有组织宣称对事件负责。"
            , 'question': "极端组织伊斯兰国是否宣称对第一起事件负责。"
            , 'answer': "是"
            , 'type': "是非"
        },
        {
            'context': "据香港媒体报道，当地时间1月7日，法国首都巴黎发生恐怖袭击，数名蒙面人袭击《沙尔利周刊》会议室，造成12人死亡，枪手仍然在逃。当局已经锁定3名枪手的身份，他们分别是34岁男子赛义德、其32岁的弟弟谢里夫以及18岁男子穆拉德，其中谢里夫曾犯有恐怖主义相关罪行。 据英国《每日电讯报》报道，赛义德与谢里夫均是法籍阿尔及利亚人，并于去年夏天由叙利亚回国。据悉，谢里夫曾是一个已被捣毁、以巴黎为基地的伊拉克圣战网络成员，并于2008年5月被控恐怖主义相关罪行，被判入狱3年，缓刑18个月。 至于另一名枪手穆拉德，其国籍仍然不明，但他居无定所，去年他入读兰斯市附近一家公立中学。"
            , 'question': "枪手穆拉德的国籍是否明确。"
            , 'answer': "还不明确"
            , 'type': "是非"
        },
        {
            'context': "据法新社报道，15日，尼日利亚东北部发生一起女性自杀式爆炸袭击造成至少7人死亡，32人受伤，这起案件被认为是博科圣地组织所为。目前，已有几十名与该组织有关的嫌疑人被捕。 这起自杀式爆炸袭击发生在尼日利亚达玛土鲁的一个公交车站，当局已在尼日尔津德尔地区逮捕了数10名激进分子嫌疑人。 报道称，一名女性在当天午后不久身缠炸药进入了汽车站，她下车之后,走向车站最里面的一个杂货店，然后停在人群中，接着引爆了自己,造成至少7人死亡,32人受伤。 一名不愿透露姓名的店主说，一群愤怒的暴徒阻止救援人员清理爆炸者的遗骸，并把那些遗骸碎片收起来一并焚烧。 虽然博科圣地组织没有宣称对此负责，但是仍被列为怀疑对象。越来越多的伊斯兰武装分子被指责在尼日利亚北部利用女人和女孩作为人体炸弹进行恐怖袭击，而公共汽车已成为首选目标。"
            , 'question': "尼日利亚东北部发生的爆炸袭击是不是由一名女性身缠炸药自爆引起的。"
            , 'answer': "是"
            , 'type': "是非"
        },

    ]
    sample = sample_list[0]
    pretrained_model_path = 'fk_t5_pegasus'
    tokenizer = T5PegasusTokenizer.from_pretrained(pretrained_model_path,
                                                   vocab_file='fk_t5_pegasus/vocab_pro.txt'
                                                   )
    pre_kn_list = fkUtils.pre_knw_list()
    tokenizer.add_special_tokens({'additional_special_tokens': pre_kn_list})

    model = Fk_MT5Model.from_pretrained(pretrained_model_path)
    model.get_device(device)

    # cques_inputs = tokenizer.encode_plus(fkUtils.remove_punctuation(sample['question']),
    #                                      fkUtils.remove_punctuation(sample['context']),
    #                                      add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
    #                                      max_length=512,
    #                                      return_tensors='pt',
    #                                      truncation=True)
    # cques_ids = cques_inputs['input_ids'].to(device)
    #
    # print(tokenizer.convert_ids_to_tokens(cques_ids[0, :]))
    # print(cques_ids.shape)

    ques_inputs = tokenizer.encode_plus(fkUtils.remove_punctuation(sample['question']),
                                        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                        max_length=512,
                                        return_tensors='pt',
                                        truncation=True)
    ques_ids = ques_inputs['input_ids'].to(device)

    text_inputs = tokenizer.encode_plus(fkUtils.remove_punctuation(sample['context']),
                                        add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                        max_length=512,
                                        return_tensors='pt',
                                        truncation=True)
    text_ids = text_inputs['input_ids'].to(device)

    ans_inputs = tokenizer.encode_plus(fkUtils.remove_punctuation(sample['answer']),
                                       add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]',
                                       max_length=512,
                                       return_tensors='pt',
                                       truncation=True)
    ans_ids = ans_inputs['input_ids'].to(device)

    model.forward(ques_ids, text_ids, ans_ids)
