"""Fast Fourier Transform using divide and conquer in time $O(n \\log n)$."""

import cmath


def fft(data):
    N = len(data)
    if N <= 1:
        return data
    data0 = data[0::2]  # even data
    data1 = data[1::2]  # odd data
    res0 = fft(data0)
    res1 = fft(data1)
    X = [0 + 0j] * N
    for k in range(N // 2):
        angle = -2 * cmath.pi * k / N
        twiddle = cmath.exp(1j * angle)
        X[k] = res0[k] + twiddle * res1[k]
        X[k + N // 2] = res0[k] - twiddle * res1[k]
    return X


if __name__ == "__main__":
    import math

    N = 2**20
    R = lambda x: (math.sin(x) + math.sin(2 * x) - math.sin(3 * x) + math.sin(x / 5) + math.cos(x)) / (0.1 * (1 + x))
    data = [R(i) for i in range(N)]
    from datetime import datetime as dt

    start = dt.now()
    fft_result = [cround(e) for e in fft(data)]
    t_fft = round((dt.now() - start).total_seconds(), 3)

    print(t_fft, "s")
