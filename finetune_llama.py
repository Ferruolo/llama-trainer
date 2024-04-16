from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer
from llama_datasets import dataset_interface
import json
import torch
from torch import cuda


MODEL_PATH = "./models/llama-7b-original/consolidated.00.pth"
PARAMS_PATH = "./models/llama-7b-original/params.json"
TOKENIZER_PATH = "./models/llama-7b-original/tokenizer.model"


BATCH_SIZE = 10
MAX_SEQ_LEN = 15
NUM_EPOCHS = 5

def main(model_path=MODEL_PATH,
         tokenizer_path=TOKENIZER_PATH,
         params_path=PARAMS_PATH,
         batch_size=BATCH_SIZE,
         max_seq_len=MAX_SEQ_LEN,
         num_epochs=NUM_EPOCHS
         ):

    if not cuda.is_available():
        print("Requires Cuda")
        exit(1)

    with open(params_path, 'r') as f:
        params = json.load(f)
        model_args = ModelArgs(
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=BATCH_SIZE,
            **params
        )

    tokenizer = Tokenizer(tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    # This is ugly and I want to make it functional but practicality wins
    # cause I'm too hungry to do it right
    models = list()
    with torch.load(model_path) as weights:
        for i in range(cuda.device_count()):
            llama = Transformer(model_args, i)
            llama.load_state_dict(weights)
            models.append(llama)

    dataset = get_grammar_dataset()


    for epoch in trange(num_epochs):
        dataset.shuffle()
        for


if __name__ == "__main__":
    fire.Fire(main)
