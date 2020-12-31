# 2-1-1. Numpy 설치
* Numpy 설치하기(Windows)
```bash
pip install numpy 
```
```python
import numpy as np
```

# 2-1-2. ndarray 개요
- **ndarray 생성**
```python
a = np.array([1, 2, 3, 4, 5]) # 1차원 배열
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]) # 2차원 배열
c = np.array([[['a', 'b'], ['c', 'd']], [['e', 'f'], ['g', 'h']]]) # 3차원 배열
d = np.array([[1, 2],[0.1, 0.2]])
```
- **ndarray.shape**: 배열의 각 차원 크기를 tuple로 반환
```python
print(a.shape, b.shape, c.shape, d.shape)

>>> (5,) (2, 4) (2, 2, 2) (2, 2)
```
- **ndarray.ndim**: 배열의 차원을 반환
```python
print(a.ndim, b.ndim, c.ndim, d.ndim)

>>> 1 2 3 2
```
- **ndarray.size**: 배열의 전체 원소의 개수를 반환
```python
print(a.size, b.size, c.size, d.size)

>>> 5 8 8 4
```
- **ndarray.dtype**: 배열 원소의 자료형을 반환
```python
print(a.dtype, b.dtype, c.dtype, d.dtype)

>>> int32 int32 <U1 float64
>>> 서로 다른 데이터 타입이 섞여 있을 경우 더 큰 데이터 타입으로 변환됨
```
- **ndarray.arange(start, end, stride)**: 범위 내의 수를 배열로 표현
```python
e = np.arange(10)
f = np.arange(10, 20, 2)
print(e)
print(f)

>>> [0 1 2 3 4 5 6 7 8 9]
>>> [10 12 14 16 18]
```
- **np.zeros(shape, dtype, order)**: 모든 값을 0으로 채운 배열 반환(order가 'C'일 경우 row 우선, 'F'일 경우 column 우선으로 채워짐)
```python
g = np.zeros((3, 2), 'int32', 'C')
h = np.zeros((2, 4), 'float64', 'F')
print(g)
print(h)

>>> [[0 0]
     [0 0]
     [0 0]]
>>> [[0. 0. 0. 0.]
     [0. 0. 0. 0.]]
```
- **np.zeros(shape, dtype, order)**: 모든 값을 1로 채운 배열 반환
```python
i = np.ones((3, 2), 'int32', 'C')
j = np.ones((2, 4), 'float64', 'F')
print(i)
print(j)

>>> [[1 1]
     [1 1]
     [1 1]]
>>> [[1. 1. 1. 1.]
     [1. 1. 1. 1.]]
```
- **np.full(shape, fill_value, dtype, order)**: 모든 값을 fill_value로 채운 배열 반환
```python
k = np.full((3, 2), 10, 'int32', 'C')
l = np.full((2, 4), 10, 'float64', 'F')
print(k)
print(l)

>>> [[10 10]
     [10 10]
     [10 10]]
>>> [[10. 10. 10. 10.]
     [10. 10. 10. 10.]]
```
- **np.reshape(shape, order)**: 배열을 shape 모양의 배열로 변환
```python
m = e.reshape(2, 5)
n = e.reshape(2, -1)
print(m)

>>> [[0 1 2 3 4]
     [5 6 7 8 9]]
>>> [[0 1 2 3 4]
     [5 6 7 8 9]]
>>> shape의 row나 col 중 하나가 -1일 경우 나머지 하나의 값에 맞춰 자동으로 변환됨(변환될 수 없을 경우 ValueError 반환)
```
- **np.ravel()**: 다차원 배열을 1차원 배열로 변환하여 반환
```python
o = m.ravel()
print(o)

>>> [0 1 2 3 4 5 6 7 8 9]
```

# 2-1-3. ndarray의 indexing
- **Fancy Indexing**
```python
a = np.arange(1, 10)
a = a.reshape(3, 3)
print(a)
print(a[[0, 1], 0:2])
print(a[[1, 2], 1:])
print(a[[0, 1]])

>>> [[1 2 3]
     [4 5 6]
     [7 8 9]]
>>> [[1 2]
     [4 5]]
>>> [[5 6]
     [8 9]]
>>> [[1 2 3]
     [4 5 6]]
```
- **Boolean Indexing**
```python
b = np.arange(1, 10)
print(b[b > 5])
print(b[b == 3])
idx = np.array([True, True, False, False, True, True, False, False, True])
print(b[idx])

>>> [6 7 8 9]
>>> [3]
>>> [1 2 5 6 9]
```

# 2-1-4. ndarray의 정렬
* np.sort(ndarray): 배열을 정렬한 뒤 반환
* ndarray.sort(): 배열을 정렬, 반환 X
```python
a = np.array([3, 6, 5, 1, 9, 2, 7, 0, 4, 8])
b = np.sort(a)
print(b)
a.sort()
print(a)

>>> [0 1 2 3 4 5 6 7 8 9]
>>> [0 1 2 3 4 5 6 7 8 9]
```

# 2-1-5. 선형대수 연산
* dot(a, b, out=None): 배열 a, b의 곱셉 결과를 반환
* vdot(a, b): 벡터 a, b의 곱셈 결과를 반환
* matmul(a, b, out=None): a, b의 행렬곱 결과를 반환
* inner(a, b): a, b의 내적 계산 결과를 반환
* outer(a, b): a, b의 외적 계산 결과를 반환
* transpose(a): a의 전치행렬 반환
* linalg.det(a): a의 행렬식 반환
* linalg.inv(a): a의 역행렬 반환
```python
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

>>> 15
>>> 39
>>> [[19 22]
     [43 50]]
>>> [[17 23]
     [39 53]]
>>> [[ 5  6  7  8]
     [10 12 14 16]
     [15 18 21 24]
     [20 24 28 32]]
>>> [[1 3]
     [2 4]]
>>> -2.0000000000000004
>>> [[-2.   1. ]
     [ 1.5 -0.5]]
```