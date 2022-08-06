from rouge import Rouge


def rouge(a, b):
    rouge = Rouge()
    rouge_score = rouge.get_scores(a, b, avg=True)  # a和b里面包含多个句子的时候用
    rouge_score1 = rouge.get_scores(a, b)  # a和b里面只包含一个句子的时候用
    # 以上两句可根据自己的需求来进行选择
    r1 = rouge_score["rouge-1"]
    r2 = rouge_score["rouge-2"]
    rl = rouge_score["rouge-l"]

    return r1, r2, rl


def main():
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
    a = []
    b = []
    for index in range(len(real_list)):
        aitem = " ".join(list(real_list[index]))
        bitem = " ".join(list(model_list[index]))

        a.append(aitem)
        b.append(bitem)

    # a = [" ".join(list("枪击事件"))," ".join(list("否"))," ".join(list("索马里青年党"))," ".join(list("还没有"))]  # 预测摘要
    # b = [" ".join(list("枪击事件"))," ".join(list("警方尚未"))," ".join(list("索马里目前"))," ".join(list("目前还没有组织或个人宣称制造"))]  # 参考摘要
    print(a)
    r1, r2, rl = rouge(a, b)
    print(r1)
    print(r2)
    print(rl)


if __name__ == '__main__':
    main()
