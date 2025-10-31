"""Use DP to minimize the raggedness of a paragraph of text."""


def break_lines(words, column=72):
    n = len(words)
    lens = [len(w) for w in words]
    dp = [(0, None)] * (n + 1)
    dp[n] = (0, None)
    for i in range(n - 1, -1, -1):
        best = (float("inf"), None)
        line_len = 0
        for j in range(i, n):
            line_len += lens[j] + (0 if j == i else 1)
            if line_len > column:
                break
            cost = 0 if j == n - 1 else (column - line_len) ** 3
            total = cost + dp[j + 1][0]
            if total < best[0]:
                best = (total, j + 1)
                dp[i] = best
                lines = []
                i = 0
    while i < n:
        j = dp[i][1]
        lines.append(" ".join(words[i:j]))
        i = j
    return lines


def justify(paragraph, column=72):
    words = paragraph.strip().split(" ")
    lines = break_lines(words, column)
    filled = []
    for line in lines[:-1]:
        chars = list(line)
        idx = 0
        while len(chars) < column:
            try:
                idx = chars.index(".", idx) + 1
                chars.insert(idx, " ")
                idx += 2
            except ValueError:
                chars.append(" ")
                idx = 0
        filled.append("".join(chars[:column]))
    filled.append(lines[-1])
    return "\n".join(filled)
