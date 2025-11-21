"""Rolling hash."""

A = 3
B = 97


def phash(word, A=3, B=97):
    n = len(word)
    return sum(ord(e) * (A ** (n - i - 1)) for i, e in enumerate(word)) % B


def window(text, k, idx, cur_hash, A, B):
    if idx == 0:
        return phash(text[:k], A, B)

    cur_hash -= ord(text[idx - 1]) * (A ** (k - 1))
    cur_hash *= A
    cur_hash += ord(text[idx + k - 1])

    return cur_hash % B


def rolling_hash(text, k):
    A = 3
    B = 97
    h = [0] * len(text)
    for i in range(len(text) - 5):
        h[i] = window(text, k, i, h[i - 1], A, B)
    return h


def find_all(h, the_hash, text, word):
    for i, e in enumerate(h):
        if e == the_hash:
            if text[i : i + len(word)] == word:
                yield i


print("hash(ALLEY) =", phash("ALLEY"))

text = """\
IN THE ALLEY BELOW THE VALLEY \
THERE'S AN ALLY CALLED MALLEY\
"""

k = 5
h = [phash(text[i : i + 5]) for i in range(len(text))]
print("h(text) =", h)
word = "ALLEY"
retval = list(find_all(h, phash(word), text, word))
print(f"find({word}) =", retval)

print(h)
print(rolling_hash(text, k))


print()
print()


tst = 0
word = "VALLEYMALLEYCALLEY"
for i in range(5):
    tst += ord(word[i]) * (A ** (k - i - 1))
    print(i, tst)
    tst %= B
    print(i, tst)
print("exp", phash(word[0:5]), word[0:5])
print("act", tst)

i = 1
print("slide")
tst -= (ord(word[0]) * A ** (k - 1)) % B
tst *= A
tst += ord(word[i + 5 - 1])
tst %= B
print("exp", phash(word[i : i + 5]), word[i : i + 5])
print(tst)
