def argmin(l):
    return min(range(len(l)), key=lambda i: l[i])


def argmax(l):
    return max(range(len(l)), key=lambda i: l[i])


def mean(l, empty_val=0):
    if len(l) == 0:
        return empty_val
    return sum(l) / len(l)


def var(l, m=None, empty_val=0):
    if len(l) == 0:
        return empty_val
    if m is None:
        m = mean(l)
    return sum([(x - m) ** 2 for x in l]) / len(l)
