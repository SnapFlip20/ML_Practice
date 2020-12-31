import numpy as np

a = np.array([1, 2, 3, 4, 5]) # 1차원 배열
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2차원 배열
c = np.array([[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]]) # 3차원 배열
d = np.array([[1, 2],[0.1, 0.2]])
print(a.shape, b.shape, c.shape, d.shape)
print(a.ndim, b.ndim, c.ndim, d.ndim)
print(a.size, b.size, c.size, d.size)
print(a.dtype, b.dtype, c.dtype, d.dtype)

print()
e = np.arange(10)
f = np.arange(10, 20, 2)
print(e)
print(f)

print()
g = np.zeros((3, 2), 'int32', 'C')
h = np.zeros((2, 4), 'float64', 'F')
print(g)
print(h)

print()
i = np.ones((3, 2), 'int32', 'C')
j = np.ones((2, 4), 'float64', 'F')
print(i)
print(j)

print()
k = np.full((3, 2), 10, 'int32', 'C')
l = np.full((2, 4), 10, 'float64', 'F')
print(k)
print(l)

print()
m = e.reshape(2, 5)
n = e.reshape(2, -1)
print(m)
print(n)

print()
o = m.ravel()
print(o)

print()
a = np.arange(1, 10)
a = a.reshape(3, 3)
print(a)
print(a[[0, 1], 0:2])
print(a[[1, 2], 1:])
print(a[[0, 1]])

print()
b = np.arange(1, 10)
print(b[b>5])
print(b[b==3])
idx = np.array([True, True, False, False, True, True, False, False, True])
print(b[idx])

a = np.array([3, 6, 5, 1, 9, 2, 7, 0, 4, 8])
b = np.sort(a)
print(b)
a.sort()
print(a)

a = np.dot(3, 5)
b = np.dot([3, 4], [5, 6])
c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])
print(a)
print(b)
print(np.matmul(c, d)) # 행렬곱
print(np.inner(c, d)) # 내적
print(np.outer(c, d)) # 외적
print(np.transpose(c)) # 전치행렬
print(np.linalg.det(c)) # 행렬식
print(np.linalg.inv(c)) # 역행렬