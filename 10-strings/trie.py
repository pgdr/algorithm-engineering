"""A basic implementation of a trie."""


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
    import sys
    import termios
    import tty

    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    DICT = "/usr/share/dict/words"
    words = []
    with open(DICT, "r") as fin:
        for line in fin:
            w = line.strip().lower()
            if w.isalnum():
                words.append(w)
    T = trie(words)
    w = ""
    inp = ""
    entered = []
    print("Enter word: ")
    while True:
        inp = getch()
        if inp == "Q":
            break
        elif ord(inp) == 13:
            entered.append(w)
            w = ""
            inp = ""
        elif ord(inp) == 127:
            inp = ""
            if w:
                w = w[: len(w) - 1]
        w = w + inp
        if w:
            for idx, x in enumerate(list(complete(T, w))):
                print(1 + idx, x)
                if idx >= 29:
                    break
        print("=" * 20)
        print(f'"{w}" (Type Q to exit, backspace to erase, enter to accept)')
    print()
    for w in entered:
        print(w)
