'''
http://ibmdecisionoptimization.github.io/docplex-doc/mp/creating_model.html
'''

from docplex.mp.model import Model
import sys

sys.path.insert(0,'/Applications/CPLEX_Studio128/cplex/python/3.6/x86-64_osx')
#print(sys.path)


myInput =[[8, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 3, 6, 0, 0, 0, 0, 0],
 [0, 7, 0, 0, 9, 0, 2, 0, 0],
 [0, 5, 0, 0, 0, 7, 0, 0, 0],
 [0, 0, 0, 0, 4, 5, 7, 0, 0],
 [0, 0, 0, 1, 0, 0, 0, 3, 0],
 [0, 0, 1, 0, 0, 0, 0, 6, 8],
 [0, 0, 8, 5, 0, 0, 0, 1, 0],
 [0, 9, 0, 0, 0, 0, 4, 0, 0]]

print("Sudoku:")
for i in range(0,9):
    for j in range(0,9):
        print(myInput[i][j],end=' ')
    print()

model = Model("sudoku")
R = range(1, 10)
idx = [(i, j, k) for i in R for j in R for k in R]

x = model.binary_var_dict(idx, name="X")

for i in R:
    for j in R:
        if myInput[i - 1][j - 1] != 0:
            model.add_constraint(x[i, j, myInput[i - 1][j - 1]] == 1)

for i in R:
    for j in R:
        model.add_constraint(model.sum(x[i, j, k] for k in R) == 1)
for j in R:
    for k in R:
        model.add_constraint(model.sum(x[i, j, k] for i in R) == 1)
for i in R:
    for k in R:
        model.add_constraint(model.sum(x[i, j, k] for j in R) == 1)

msol = model.solve()

# Minimize cost
# mdl.minimize(mdl.sum(qty[f] * f.unit_cost for f in foods))
# mdl.print_information()
# mdl.export_as_lp()

if msol:
    print("Solution:")
    for i in R:
        for j in R:
            for k in R:
                if msol.get_var_value(x[(i,j,k)]) == 1:
                    print(str(k),end=' ')
        print()
else:
    print("Solve status: " + msol.get_solve_status() + "\n")