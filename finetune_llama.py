from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer
from llama_datasets.dataset_interface import get_dataset
import json
import torch
from torch import cuda
import fire

MODEL_PATH = "./models/llama-7b-original/consolidated.00.pth"
PARAMS_PATH = "./models/llama-7b-original/params.json"
TOKENIZER_PATH = "./models/llama-7b-original/tokenizer.model"
DATASET_NAME = "grammar"

BATCH_SIZE = 10
MAX_SEQ_LEN = 15
NUM_EPOCHS = 5


def main(model_path=MODEL_PATH,
         tokenizer_path=TOKENIZER_PATH,
         params_path=PARAMS_PATH,
         batch_size=BATCH_SIZE,
         max_seq_len=MAX_SEQ_LEN,
         num_epochs=NUM_EPOCHS,
         dataset_name=DATASET_NAME
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

    models = list()
    for device_num in range(cuda.device_count()):
        weights = torch.load(model_path)
        llama = Transformer(model_args, device_num)
        llama.load(weights)
        models.append(llama)


    (train, val) = get_dataset(dataset_name, tokenizer)


if __name__ == "__main__":
    fire.Fire(main)
