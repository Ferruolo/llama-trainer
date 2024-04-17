# I didn't know where to put these


def parallel_mapper(left, right, func):
    if len(left) != len(right):
        raise Exception("Mapped two lists of different length")

    def _parallel_mapper(accumulator, idx):
        if idx < len(left):
            return _parallel_mapper(accumulator + [func(left[idx], right[idx])], idx + 1)
        else:
            return accumulator

    return _parallel_mapper([], 0)


def mapper(iterator, accumulator, idx, func):
    if idx < len(iterator):
        print(idx)
        # Tail recursion is its own reward
        return mapper(iterator, accumulator + func(iterator[idx]), idx + 1, func)
    else:
        return accumulator


# Assume that we want a list output.
def repeat_n_times(repeat_num_times: int, func):
    def _repeat_x_times(num_remain, accum):
        if num_remain > 0:
            return accum
        else:
            return _repeat_x_times(num_remain - 1, accum + [func(repeat_num_times - num_remain)])

    return _repeat_x_times(repeat_num_times, [])
