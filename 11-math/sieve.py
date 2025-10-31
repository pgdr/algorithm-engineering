"""The Sieve of Eratosthenes — an ancient algorithm for finding all prime numbers up to any given limit."""


def sieve(n):
    if n < 2:
        return
    prime = [True] * (n + 1)
    prime[0] = False
    prime[1] = False
    for i in range(2, n + 1):
        if not prime[i]:
            continue
        yield i
        for j in range(i + i, n, i):
            prime[j] = False


print(*sieve(101))
