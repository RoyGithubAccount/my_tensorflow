"""

Program to multiply two matrices with a nested for list

we will use zip() and with the * operator

"""

# 3 x 3 matrix

X = [[12,7,3],
     [4, 5, 6],
     [7 , 8, 9]]

# 3 x 4 matrix
Y = [[5, 8, 1, 2],
     [6, 7, 3, 0],
     [4, 5, 9, 1]]

# result is 3x4 matrix

result = [[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]

for r in result:
    print(r)

