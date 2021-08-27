from itertools import permutations, product


def dictionary(attributes, values):

    d = {a: [] for a in attributes}
    for i, v in enumerate(values):
        d[attributes[i]].append(v)
    return d


def flatten(li):
    def deep(li):

        for el in li:
            if isinstance(el, (list, tuple)):
                deep(el)
            else:
                flat_list.append(el)
        return None

    flat_list = []
    deep(li)
    flat_list = [
        (flat_list[2 * i], flat_list[2 * i + 1])
        for i in range(int(len(flat_list) / 2))
    ]
    return flat_list


def get_combinations(set1, set2):

    pairs = [list(product([el], set2)) for el in set1]
    combinations = [
        [pairs[i][p[i]] for i in range(len(p))]
        for p in permutations(range(len(pairs)))
    ]
    return combinations


def group_by(iterable, key=lambda x: (x, x)):

    items = [key(el) for el in iterable]
    groups = {k: [] for k, v in items}
    [groups[k].append(v) for k, v in items]
    return groups


def remove_items(dictionary, cond):

    removed_items = []
    for item in list(dictionary.items()):
        if cond(item):
            dictionary.pop(item[0])
            removed_items.append(item[1])
    return removed_items


def calc_gcd(a, b):

    while a != 0 and b != 0:
        if a > b:
            a %= b
        else:
            b %= a
    return a + b
