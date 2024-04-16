import pathlib
from torch.utils.data import Dataset
import csv
from grammar_dataset.process import preprocess_grammar_dataset, get_grammar_idx, load_grammar_data, get_grammar_len

current_folder = pathlib.Path(__file__).parent.resolve().__str__()

## A working typesystem would be really sweet right about here
dataset_tools = {
    "grammar": (
        (current_folder + "/gtrain_10k.csv", current_folder + "/grammar_validation.csv"),
        (preprocess_grammar_dataset, get_grammar_idx, load_grammar_data, get_grammar_len)
    )
}


class dataloader(Dataset):
    def __init__(
            self,
            tokenizer,
            file_path,
            processor,
            getter,
            loader,
            get_len
    ):
        # The huggingface boycott continues, making my own dataloader here
        # Assume for now that dataset is small enough that we don't need to complicate things
        # also assume that we don't need a validation set
        # ((train_path, test_pat), (processor, getter)) = dataset_paths[dataset]
        self.datastore = loader(file_path, processor)
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.getter = getter
        self.loader = loader
        self.get_len = len
        self.print_text = False  # print_text

    def __len__(self):
        return self.get_len(self.datastore)

    def __getitem__(self, index):
        (input_, target_) = self.getter(self.datastore, index)

        prompt_ids = self.tokenizer.encode(input_, add_special_tokens=False)
        label_ids = self.tokenizer.encode(target_, add_special_tokens=False)

        sample = {
            "input_ids": prompt_ids + label_ids,
            "attention_mask": [1] * len(prompt_ids + label_ids),
            "labels": [-100] * len(prompt_ids) + label_ids
        }

        return sample


def get_dataset(dataset_name, tokenizer):
    ((train_path, test_path), (processor, getter, loader, get_len)) = dataset_tools[dataset_name]
    trainer = dataloader(tokenizer, train_path, processor, getter, loader, get_len)
    tester = dataloader(tokenizer, test_path, processor, getter, loader, get_len)
    return trainer, tester
