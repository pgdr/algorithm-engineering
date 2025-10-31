"""Maximum consecutive sum in an array with possibly negative values."""


def consecutive_sum(array, idx=0, con_sum=0, global_max=0):
    if idx >= len(array):
        return global_max
    copt = max(array[idx], array[idx] + con_sum)
    gopt = max(global_max, copt)
    return consecutive_sum(array, idx + 1, copt, gopt)


print(consecutive_sum([4, 0, 3, -8, 1, 4, -2, -10]))
