import csv
import random

INPUT_START_LABEL = "[ISTART]"
INPUT_END_LABEL = "[IEND]"
OUTPUT_START_LABEL = "[OSTART]"
OUTPUT_END_LABEL = '[OEND]'


def preprocess_grammar_dataset(row):
    input_, target_ = row
    wrapped_input = INPUT_START_LABEL + input_ + INPUT_END_LABEL
    wrapped_output = OUTPUT_START_LABEL + target_ + OUTPUT_END_LABEL

    return (wrapped_input, wrapped_output)


def get_grammar_idx(dataset, idx):
    return dataset[int(idx)]


def load_grammar_data(filepath: str, processor) -> list:
    datalist = list()
    with open(filepath) as f:
        csvfile = csv.reader(f)
        for row in csvfile:
            datalist.append(processor(row))
    return datalist


def get_grammar_len(datastore):
    return len(datastore)


def shuffle_dataset(data_list):
    random.shuffle(data_list)