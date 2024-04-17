from llama_datasets.dataset_interface import distributed_dataloader as dataloader
from llama import Transformer as llama_model # Import type for analyzer
from configure.config import TrainerArgs
from training_lib.synchronizer import Synchronizer, sync_enums
from training_lib.gradient_updates import gradient_update


def training_thread(
        model: llama_model,
        device_id,
        training_config: TrainerArgs,
        training_dataloader: dataloader,
        test_dataloader: dataloader,
        synchronizer: Synchronizer
):
    model.cuda(device_id)
    optimizer = training_config.optimizer(model.parameters(), lr=training_config.learning_rate)
    loss_func = training_config.loss_func()

    while synchronizer.await_receive(device_id) is not sync_enums.stop:

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
            model = gradient_update(model, device_id, synchronizer)
            optimizer.step()

            epoch_loss += loss.item()
            model.eval()


        (x_test, y_test) = test_dataloader.get_full(device_id)
        y_pred = model.forward(x_test, 0)
        val_loss = loss_func(y_pred, y_test).item()

        synchronizer.send_to_host((epoch_loss, val_loss))
    return model
