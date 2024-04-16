import pathlib
import torch
from llama_datasets.grammar_dataset.process import preprocess_grammar_dataset, get_grammar_idx, load_grammar_data, \
    get_grammar_len, shuffle_dataset
from torch import cuda

current_folder = pathlib.Path(__file__).parent.resolve().__str__()

## A working typesystem would be really sweet right about here
dataset_tools = {
    "grammar": (
        (
            current_folder + "/grammar_dataset/gtrain_10k.csv",
            current_folder + "/grammar_dataset/grammar_validation.csv"),
        (preprocess_grammar_dataset, get_grammar_idx, load_grammar_data, get_grammar_len)
    )
}


# Had to do this myself :(
## Might not be abstract enough???
class distributed_dataloader:
    def __init__(
            self,
            tokenizer,
            batch_size,
            file_path,
            processor,
            getter,
            loader,
            get_len,
            shuffle
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
        self.get_len = get_len
        self.print_text = False  # print_text
        self.batch_size = batch_size
        self._shuffle = shuffle
        self.gpu_store = list()

    def len(self):
        return self.get_len(self.datastore)

    def shuffle(self):
        self._shuffle(self.datastore)

    def prep_next_epoch(self):
        self.shuffle()
        items_per_gpu = self.len() // cuda.device_count()
        total_index = 0
        for gpu_num in range(cuda.device_count()):
            batch_index = 0
            batch = list()
            max_len_x = 0
            max_len_y = 0
            batch_x = list()
            batch_y = list()
            print(items_per_gpu)
            for item_idx in range(items_per_gpu):
                print(item_idx)
                ## Complete this code
                if batch_index < self.batch_size:
                    (x, y) = self.get_item(total_index)
                    max_len_x = max(max_len_x, len(x))
                    max_len_y = max(max_len_y, len(y))
                    batch_x.append(x)
                    batch_y.append(y)
                    batch_index += 1
                    total_index += 1
                else:
                    batch_x = [self.pad(item, max_len_x) for item in batch_x]
                    batch_y = [self.pad(item, max_len_y) for item in batch_y]


                    tensor_x = torch.Tensor(batch_x).cuda(gpu_num)
                    tensor_y = torch.Tensor(batch_y).cuda(gpu_num)
                    batch.append((tensor_x, tensor_y))
                    batch_x, batch_y = list(), list()
                    batch_index = 0

            self.gpu_store.append(batch)

    def get_next_batch(self, gpu_idx):
        # Implement me:
        # This function gets the next batch for a certain GPU
        item = self.gpu_store[gpu_idx].pop()
        return item

    def get_item(self, index) -> [int]:
        (input_, target_) = self.getter(self.datastore, index)

        prompt_ids = self.tokenizer.encode(input_, False, False)
        label_ids = self.tokenizer.encode(target_, False, False)

        sample = (
            prompt_ids, label_ids
        )

        return sample

    def pad(self, item: list, max_len):
        while len(item) < max_len:
            item.append(self.tokenizer.pad_id)
        return item


def get_dataset(dataset_name, tokenizer, batch_size):
    ((train_path, test_path), (processor, getter, loader, get_len)) = dataset_tools[dataset_name]


    trainer = distributed_dataloader(tokenizer, batch_size, train_path, processor, getter, loader, get_len,
                                     shuffle_dataset)
    tester = distributed_dataloader(tokenizer, batch_size, test_path, processor, getter, loader, get_len,
                                    shuffle_dataset)
    return trainer, tester
