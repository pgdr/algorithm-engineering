"""Sliding window to find best $k$ sized window in linear time."""

data = [3.5, 7.5, 8.0, 5.7, 3.1, 4.2, 7.2, 0.1, 3.4, 1.2, -4]
k = 3
acc = sum(data[:k])
best = acc
winstart = 0

for winstart in range(1, len(data) - k + 1):
    acc += -data[winstart - 1] + data[winstart + k - 1]
    if acc > best:
        best = acc

print(best)
