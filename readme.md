# LLama Trainer

Custom setup for training llama.
Can't afford to use huggingface format as I need pytorch flexibility
for future endevors with these weights.

Trying to make this as flexible/simple as possible for reuse

# Weight Sync
Taking advantage of the parallelism, I sync weights as follows in traininglib/gradient_updates.py

1. Initialize divisor as 2
2. if item index is not divisible by divisor, send gradients to device_num // divisor
3. add the two gradients
4. repeat until we get to one one gradient remaining
5. average, and send everything back

Note that the accumulation process here takes log(num_gpu) iterations, which is important because gpu-gpu transfer is expenseive


