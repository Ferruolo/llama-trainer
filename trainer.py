# Adapting a similar approach to the end goal 
# for Stainless!

# No MSPC library here (what a sham) so we have to re-invent it using a dictionary
from torch import cuda
from multiprocessing import Process, Queue



def distributed_trainer(
        model_list, 
        training_set, 
        test_set,
        training_func
        ):
    print("======Began Training=======")

    #Kinda a hacky Multi-Producer Single Consumer implementation
    # but its what we got here
    synchronizer = list()

    threads = list()
    for i in cuda.device_count():
        sender = Queue()
        reciever = Queue()
        
        training_process = Process(target=training_func, args=(
            model_list[i],
            training_config, 
            i,
            training_set,
            test_set

            ))


    total_loss = 0






