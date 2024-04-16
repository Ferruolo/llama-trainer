
INPUT_START_LABEL = "[ISTART]"
INPUT_END_LABEL = "[IEND]"
OUTPUT_START_LABEL = "[OSTART]"
OUTPUT_END_LABEL = '[OEND]'



def preprocess_grammar_dataset(row):
    input_, target_ = row
    wrapped_input  = INPUT_START_LABEL + input_ + INPUT_END_LABEL
    wrapped_output = OUTPUT_START_LABEL + output_ + OUTPUT_END_LABEL

    return (wrapped_input, wrapped_output)



def get_grammar_idx(dataset, idx):
    

    return dataset[int(idx)]

def load_grammar_data(filepath: str, processor) -> list:
    datalist = list()
    with open(path_to_data_file) as f:
        csvfile = csv.reader(f)
        for row in csvfile:
            self.datastore.append(processor(row))
    return dataset



def get_grammar_len(datastore):
    return len(datastore)


