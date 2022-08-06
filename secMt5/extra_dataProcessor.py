import os
import time
import json
import random
import datetime
import numpy as np
from tqdm import tqdm
from transformers import (
    DataProcessor,
    BertTokenizer,
    squad_convert_examples_to_features,
    BertForQuestionAnswering,
)

class MySquadProcessor(DataProcessor):
    train_file = "train.json"
    dev_file = "dev.json"
    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                qas_id = paragraph["id"]
                qas_list = paragraph["qas"]


                context_text = paragraph["context"]
                question_text = paragraph["GTquestion"]
                start_position_character = None
                answer_text = paragraph["GTanwser"]
                answers = []
                is_impossible = True

                for qa_pair in qas_list:
                    cq_question_text = qa_pair["question"]
                    cq_answer_text = qa_pair["answers"][0]["text"]
                    cq_example = MyChineseExample(
                        qas_id=qas_id,
                        question_text=cq_question_text,
                        context_text=context_text,
                        answer_text=cq_answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(cq_example)

                example = MyChineseExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    title=title,
                    is_impossible=is_impossible,
                    answers=answers,
                )

                examples.append(example)

        return examples

class MyChineseExample(object):
    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=True,
    ):

        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text.replace(" ","").replace("  ","").replace(" ","").replace("\n","")
        self.answer_text = str(answer_text)
        # print('question_text == ',question_text)
        # print('answer_text == ',answer_text)


        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers
        self.doc_tokens = [e for e in self.context_text]
        self.char_to_word_offset = [i for i, e in enumerate(self.context_text)]
        anstr = str(answer_text).replace(" ","").replace(" ","")
        self.start_position = self.context_text.find(anstr)
        self.end_position = self.start_position + len(anstr)


if __name__ == '__main__':
    data_dir = './dataset/'
    processor = MySquadProcessor()
    Train_data = processor.get_train_examples(data_dir)