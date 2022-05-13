from simplex import solve, parse_file, build_dual

testfile = 'test2.txt'

A, b, c, n, indexes_not_bounded, mode = parse_file(testfile)

print('Problem solution:')
sol = solve(A, b, c, n, indexes_not_bounded, mode, 'bruteforce')
if sol:
    x, v = sol
    print(f'Bruteforce method: x = {x}, v = {v}')
else:
    print('Bruteforce: No solution')

sol = solve(A, b, c, n, indexes_not_bounded, mode, 'table')
if sol:
    x, v = sol
    print(f'Table method: x = {x}, v = {v}')
else:
    print('Table method :No solution')

A, b, c, n, indexes_not_bounded, mode = build_dual(A, b, c, n, indexes_not_bounded, mode)
print('Dual solution:')
sol = solve(A, b, c, n, indexes_not_bounded, mode, 'bruteforce')
if sol:
    x, v = sol
    print(f'Bruteforce method: x = {x}, v = {v}')
else:
    print('Bruteforce: No solution')

sol = solve(A, b, c, n, indexes_not_bounded, mode, 'table')
if sol:
    x, v = sol
    print(f'Table method: x = {x}, v = {v}')
else:
    print('Table method :No solution')