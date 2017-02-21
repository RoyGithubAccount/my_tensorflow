"""

Program to multiply two matrices with a nested for loop

"""

# 3 x 3 matrix

X = [[12,7,3],
     [4, 5, 6],
     [7 , 8, 9]]

print("X =", X )

# 3 x 4 matrix
Y = [[5, 8, 1, 2],
     [6, 7, 3, 0],
     [4, 5, 9, 1]]

print("Y =", Y)

# result will be a 3 x 4 matrix X's row x Y's column. Here we initialise it
result = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

print("len(X)", len(X), "The number of lists which implies rows")
print("for i in range(len(X)): is the outer iterator, has 3 rows so this should be 3")
print("")
print("len(Y[0]", len(Y[0]), "The number of elements in a list which is also a row")
print("for j in range(len(Y[0])): is the items in Y's row so this should be 4")
print("")
print("len(Y)", len(Y), "The number of lists which implies rows")
print("for k in range(len(Y)): is the inner most iterator, has 3 rows so this should be 3")
print("")
print("result[i][j] += X[i][k] * Y[k][j]")

# iterate through rows of X
for i in range(len(X)):
    # iterate through columns of Y
    for j in range(len(Y[0])):
        # iterate through rows of Y
        for k in range(len(Y)):
            result[i][j] += X[i][k] * Y[k][j]

for r in result:
    print(r)
