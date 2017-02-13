import numpy as np
tensor_3d = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
print(tensor_3d)
print ("shape ", tensor_3d.shape)

print("visualise it as two 2 x 2 matrix stacked on top of each other")

print("Retrieve element [0,0,0] - top 2x2, 0 row, 0 col = 1 ") 
print(tensor_3d[0,0,0])


print("Retrieve element [0,0,1] - top 2x2, 0 row, 1 col = 2")
print(tensor_3d[0,0,1])

print("Retrieve element [0,1,0] - top 2x2, 1 row, 0 col = 3")
print(tensor_3d[0,1,0])

print("Retrieve element [0,1,1] - top 2x2, 1 row, 1 col = 4")
print(tensor_3d[0,1,1])

print("Retrieve element [1,0,0] - bottom 2x2, 0 row, 0 col = 5")
print(tensor_3d[1,0,0])

print("Retrieve element [1,0,1] - bottom 2x2, 0 row, 1 col = 6")
print(tensor_3d[1,0,1])

print("Retrieve element [1,1,0] - bottom 2x2, 1 row, 0 col = 7")
print(tensor_3d[1,1,0])

print("Retrieve element [1,1,1] - bottom 2x2, 1 row, 1 col = 8")
print(tensor_3d[1,1,1])
