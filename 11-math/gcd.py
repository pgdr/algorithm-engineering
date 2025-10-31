"""Extended Euclidean algorithm: compute the coefficients of Bézout's identity, i.e., $ax + by = \\gcd(a, b)$."""


def gcdx(a, b):
    if a == 0:
        return b, 0, 1
    g, x1, y1 = gcdx(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return g, x, y


A = 450
B = 1080
g, x, y = gcdx(450, 1080)
if y >= 0:
    print(f"{A}·{x} + {B}·{y} = gcd({A}, {B}) = {g}")
else:
    print(f"{A}·{x} - {B}·{-y} = gcd({A}, {B}) = {g}")
