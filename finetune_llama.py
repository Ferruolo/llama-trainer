import json
import torch
from torch import cuda
import fire

from configure.config import ModelArgs, TrainerArgs
from llama.tokenizer import Tokenizer
from llama_datasets.dataset_interface import get_dataset
from training_lib.distributed_trainer import distributed_trainer
from llama import Transformer

MODEL_PATH = "./models/llama-7b-original/consolidated.00.pth"
PARAMS_PATH = "./models/llama-7b-original/params.json"
TOKENIZER_PATH = "./models/llama-7b-original/tokenizer.model"
DATASET_NAME = "grammar"
SAVE_PATH = "./models/finetuned.pth"


BATCH_SIZE = 10
MAX_SEQ_LEN = 15
NUM_EPOCHS = 5


def main(model_path=MODEL_PATH,
         tokenizer_path=TOKENIZER_PATH,
         params_path=PARAMS_PATH,
         batch_size=BATCH_SIZE,
         max_seq_len=MAX_SEQ_LEN,
         num_epochs=NUM_EPOCHS,
         dataset_name=DATASET_NAME,
         save_path=SAVE_PATH
         ):

    if not cuda.is_available():
        print("Requires Cuda")
        exit(1)
    print("Cuda Available: Commencing Training")
    with open(params_path, 'r') as f:
        params = json.load(f)
        model_args = ModelArgs(
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=BATCH_SIZE,
            **params
        )
    training_config = TrainerArgs()
    tokenizer = Tokenizer(tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    print("Accessories Loaded")

    weights = torch.load(model_path)
    llama = Transformer(model_args, 0)
    llama.load(weights)
    print("Model Loaded")
    (train, val) = get_dataset(dataset_name, tokenizer, batch_size=training_config.batch_size)
    print("Data Loaded")
    llama = distributed_trainer(llama, training_config, train, val)
    llama = llama.to('cpu')
    torch.save(llama.state_dict(), save_path)


if __name__ == "__main__":
    fire.Fire(main)
