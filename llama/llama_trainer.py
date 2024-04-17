from torch import optim
from torch.nn import CrossEntropyLoss
from llama_datasets.dataset_interface import distributed_dataloader as dataloader
from llama import Transformer as llama_model # Import type for analyzer
from enums import sync_enums



class TrainerArgs:
    learning_rate = 1e-4
    optim_params = {}
    optimizer = optim.Adam
    loss_func = CrossEntropyLoss
    num_epochs = 5


# like a lion tamer, but for llamas
def llama_tamer(
        model: llama_model,
        device_id,
        getter,
        setter,
        training_config: TrainerArgs,
        training_dataloader: dataloader,
        test_dataloader: dataloader):

    model.cuda(device_id)
    optimizer = training_config.optimizer(model.parameters(), lr=training_config.learning_rate)
    loss_func = training_config.loss_func()

    while True:
        continue_training = getter()
        if continue_training is not sync_enums.stop:
            break

        epoch_loss = 0
        next_batch = training_dataloader.get_next_batch(device_id)
        model.train()
        # I wish I was in rust, or ocaml, or some other functional language
        while next_batch is not None:
            optimizer.zero_grad()
            (x, y_true) = next_batch

            y_pred = model.forward(next_batch, 0)
            loss = loss_func(y_pred, y_true)
            loss.backward()

            # Sync Gradients here (unless we can sync this more efficiently)!!!!
            while


            optimizer.step()
            epoch_loss += loss.item()
            model.eval()

        setter(sync_enums.completed_epoch)

        (x_test, y_test) = test_dataloader.get_full(device_id)
        y_pred = model.forward(x_test, 0)
        val_loss = loss_func(y_pred, y_test)

        setter(val_loss)

    return model
