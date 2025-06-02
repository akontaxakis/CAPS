A = {
    't0': {'name': 't0', 'cost': 1, 'tail': ['s'],        'head': ['v0']},
    't1': {'name': 't1', 'cost': 1, 'tail': ['v0'],       'head': ['v1', 'v2']},
    't2': {'name': 't2', 'cost': 1, 'tail': ['v1'],       'head': ['v3', 'v4']},
    't3': {'name': 't3', 'cost': 1, 'tail': ['v2', 'v4'], 'head': ['v5']},
    't4': {'name': 't4', 'cost': 1, 'tail': ['v3'],       'head': ['v6']},
    't5': {'name': 't5', 'cost': 1, 'tail': ['v3', 'v6'], 'head': ['v7']},
    't6': {'name': 't6', 'cost': 1, 'tail': ['v5', 'v6'], 'head': ['v8']},
    't7': {'name': 't7', 'cost': 1, 'tail': ['v1'],       'head': ['v3', 'v4']},
    'l1': {'name': 'l1', 'cost': 1, 'tail': ['s'],        'head': ['v1']},
    'l2': {'name': 'l2', 'cost': 1, 'tail': ['s'],        'head': ['v2']},
    'l3': {'name': 'l3', 'cost': 1, 'tail': ['s'],        'head': ['v3']},
    'l4': {'name': 'l4', 'cost': 1, 'tail': ['s'],        'head': ['v4']},
    }


def CartesianProduct(sets):
    if len(sets) == 0:
        return [[]]
    else:
        CP = []
        current = sets.popitem()
        for c in current[1]:
            for set in CartesianProduct(sets):
                CP.append(set+[c])
        sets[current[0]] = current[1]
        return CP


def bstar(A, v):
    edges = []
    for e in A.values():
        if v in e['head']:
            edges.append(e)
    return edges


def Expand(A, pi, s):
    PI = []
    E = {}
    for v in [v_prime for v_prime in pi['frontier'] if v_prime not in s]:
        E[v] = bstar(A, v)
    M = CartesianProduct(E)
    for move in M:
        pi_prime = {
            'cost': pi['cost'],
            'visited': pi['visited'].copy(),
            'frontier': [],
            'plan': pi['plan'].copy()
            }
        for e in move:
            new_nodes = [n for n in e['head'] if n not in pi_prime['visited']]
            if new_nodes:
                pi_prime['cost'] += e['cost']
                pi_prime['plan'].append(e)
                pi_prime['visited'] += new_nodes
                pi_prime['frontier'] += [n for n in e['tail'] if n not in (pi_prime['visited']+pi_prime['frontier'])]
        PI.append(pi_prime)
    return PI


def Optimize(A, s, T):
    cost_star = 999999
    pi_star = []
    Q = [{'cost': 0, 'visited': [], 'frontier': T, 'plan': []}]
    while Q:
        pi = Q.pop(0)
        if pi['cost'] <= cost_star:
            if pi['frontier'] == s:
                cost_star = pi['cost']
                pi_star.append(pi)
            else:
                for pi_prime in Expand(A, pi, s):
                    Q.append(pi_prime)
    return pi_star


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(Optimize(A, ['s'], ['v7', 'v8']))