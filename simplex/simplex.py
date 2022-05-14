import numpy as np

from .simplex_table import simplex_table
from .ext_points_search import solve_brute_force


def parse_file(filename):
    n = 0
    mode = 'max'
    A = []
    b = np.array([])
    indexes_not_bounded = []
    with open(filename, 'r') as file:
        for line in file:
            if '=' not in line:
                indexes_not_bounded = {i for i in range(n)} - {int(x) - 1 for x in line.split()}
                continue
            sep = '='
            if '<' in line:
                sep = '<='
            elif '>' in line:
                sep = '>='
            line_split = line.split(sep)
            if len(line_split) != 2:
                return None
            lhs, rhs = line_split
            lhs = lhs.strip().lower()
            rhs = rhs.strip().lower()
            if lhs == 'n':
                if not rhs.isdigit() or int(rhs) <= 0 or sep != '=':
                    return None
                n = int(rhs)
            elif lhs == 'mode':
                mode = rhs
                if rhs not in ('min', 'max') or sep != '=':
                    return None
            elif lhs == 'c':
                if sep != '=':
                    return None
                c = np.array([float(coef) for coef in rhs.split()])
            else:
                a = np.array([float(coef) for coef in lhs.split()])
                bi = float(rhs)
                if sep == '>=':
                    a *= -1
                    bi *= -1
                elif sep == '=':
                    A.append(-a)
                    b = np.append(b, -bi)
                A.append(a)
                b = np.append(b, bi)

    A = np.array(A)
    for i in indexes_not_bounded:
        c = np.append(c, -c[i])
        A = np.append(A, -A.take(i, axis=1).reshape(len(b), 1), axis=1)

    return A, b, c, n, indexes_not_bounded, mode


def build_dual(A, b, c, n, indexes_not_bounded, mode):
    if mode.lower() == 'max':
        return -A.T, -c, -b, n, indexes_not_bounded, 'max'
    else:
        return -A.T, c, -b, n, indexes_not_bounded, 'max'


def solve(A, b, c_in, n, indexes_not_bounded, mode, method='table'):
    if mode.lower() == 'min':
        c = -1 * c_in
    else:
        c = c_in
    if method == 'table':
        sol = simplex_table(A, b, c)
    elif method == 'bruteforce':
        sol = solve_brute_force(A, b, c)
    else:
        raise 'Choose correct method'

    if not sol:
        return None
    else:
        x, v = sol

    if mode == 'min':
        v = -v

    for j, i in enumerate(indexes_not_bounded):
        x[i] = x[i] - x[n + j]
    x = x[:n]

    return x, v
