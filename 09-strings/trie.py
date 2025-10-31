def trie(words):
    root = {}
    for word in words:
        current = root
        for letter in word:
            current = current.setdefault(letter, {})
        current["*"] = "*"
    return root


def find(T, w):
    current = T
    for letter in w:
        if letter not in current:
            return None  # Not found
        current = current[letter]
    return current  # Returns node in tree


def _dfs(T):
    for k in T:
        if k == "*":
            yield ""
        else:
            for v in _dfs(T[k]):
                yield k + v


def complete(T, w):
    C = find(T, w)  # continuations
    if not C:
        return []
    for c in _dfs(C):
        yield w + c


if __name__ == "__main__":
    from pprint import pprint as print

    words = ["car", "can", "cart", "cat", "cup", "dent", "denim", "do", "dog", "dot"]
    T = trie(words)
    print(find(T, "do"))
    print(find(T, "de"))
    print(list(complete(T, "do")))
    print(list(complete(T, "de")))
