x =[1,2,3]
print("raw x", x)
y = [4,5,6]
print("raw x", y)
zipped = zip(x,y)
new_matrix =list(zipped)
print("Zipped x & y", new_matrix)
# [(1, 4), (2, 5), (3, 6)]
x2, y2 = zip(*zip(x,y))
print("Unzipped x part in x2", x2)
# (1, 2, 3)
print("Unzipped y part in y2", y2)
#(4, 5, 6)
if x == list(x2) and y ==list(y2):
    print("True")
else:
    print("False")
# True
k = list(range(3,6))
print(k)
args =[3,6]
print("Before unpacking", args)
j = list(range(*args))  # the * operator breaks the args list in to two variables passed in to list
print("After unpacking", j)

# merge two dicts
w = {'a':1, 'b':2}
print("w", w)
v = {'b':3, 'c':4}
print("v", v)
z = {**w, **v}
print("z = {**w, **v)", z, "   Note b from last dict to be passed overwirtes earlier b = 2")