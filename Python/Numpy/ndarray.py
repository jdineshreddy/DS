import numpy as np
a  = np.arange(10)

print(a, a.ndim, a.shape, type(a), type(a).__name__, a.dtype, a.itemsize,a.nbytes, a.size, a.data)
#[0 1 2 3 4 5 6 7 8 9] 1 (10,) <class 'numpy.ndarray'> ndarray int32 4 40 10 <memory at 0x0000025FD77BCF48>

b = a.reshape(2,5)

print(b)     #[[0 1 2 3 4]
             #[5 6 7 8 9]]

print(b[0])    #[0 1 2 3 4]
print(b[1])    #[5 6 7 8 9]

print(b[:])   #[[0 1 2 3 4]
               #[5 6 7 8 9]]

print(b[1,3]) # 8

print(b[:,3])  # [3 8] prints 4th column
print(b[:,6])  # index 6 is out of bounds for axis 1 with size 5
print(b[:,1])  # [1,6] prints 2nd column

c = b[:,1]

c  # array([1, 6])
c[0] = 4  # this also changes the corresponding element in b
c  # array([4, 6])
b  # array([[0, 4, 2, 3, 4],
           #[5, 6, 7, 8, 9]])

c # array([4, 6])
c[0] , c[1] = 8,9
c # array([8, 9])
b #array([[0, 8, 2, 3, 4],
         #[5, 9, 7, 8, 9]])

c[3] = 7 #  index 3 is out of bounds for axis 0 with size 2

a         # array([0, 8, 2, 3, 4, 5, 9, 7, 8, 9])
d = a.T   # array([0, 8, 2, 3, 4, 5, 9, 7, 8, 9])

b   # array([[0, 8, 2, 3, 4],
      #      [5, 9, 7, 8, 9]])

e = b.T

e   #array([[0, 5],
     #  [8, 9],
     #  [2, 7],
     #  [3, 8],
     #  [4, 9]])

b.flatten() # array([0, 8, 2, 3, 4, 5, 9, 7, 8, 9])

# order : {‘C’, ‘F’}, optional
# Row-major (C-style) or column-major (Fortran-style) order
f =  np.array([[0, 1], [2, 3]], order='C')  # array([[0, 1],
                                                    #[2, 3]])
#f[0] # array([0, 1])
f.resize((2,1))  # array([[0],
                         #[1]])

g =  np.array([[0, 1], [2, 3]], order='F')  #array([[0, 1],
                                                   #[2, 3]])
#g[0] # array([0, 1])
g.resize((2,1))     # array([[0],
                           # [2]])


h = np.array([[1,2],[3,4]])
h.resize(2,3)   #array([[1, 2, 3],
                       #[4, 0, 0]])


x = np.zeros((3, 4, 5))

#array([[[ 0.,  0.,  0.,  0.,  0.],
 #       [ 0.,  0.,  0.,  0.,  0.],
  #      [ 0.,  0.,  0.,  0.,  0.],
   #     [ 0.,  0.,  0.,  0.,  0.]],
#
 #      [[ 0.,  0.,  0.,  0.,  0.],
  #      [ 0.,  0.,  0.,  0.,  0.],
   #     [ 0.,  0.,  0.,  0.,  0.],
    #    [ 0.,  0.,  0.,  0.,  0.]],
#
 #      [[ 0.,  0.,  0.,  0.,  0.],
  #      [ 0.,  0.,  0.,  0.,  0.],
   #     [ 0.,  0.,  0.,  0.,  0.],
    #    [ 0.,  0.,  0.,  0.,  0.]]])
#numpy.moveaxis(a, source, destination)[source]
np.moveaxis(x, 0, -1).shape  # (4, 5, 3)


y = np.ones((4,2,3))
#array([[[ 1.,  1.,  1.],
 #       [ 1.,  1.,  1.]],
#
 #      [[ 1.,  1.,  1.],
  #      [ 1.,  1.,  1.]],
#
 #      [[ 1.,  1.,  1.],
  #      [ 1.,  1.,  1.]],
#
 #      [[ 1.,  1.,  1.],
  #      [ 1.,  1.,  1.]]])
np.moveaxis(y, 0, -1).shape  # (2, 3, 4)

z = np.ones((3,4,5,6))

k = np.array(([1,2],[3,4]), np.uint8)

k[0] # array([1, 2], dtype=uint8)

k[0] = 3
 
k # array([[3, 3],
       #    [3, 4]], dtype=uint8)

k[0] = -998  # array([[25, 25],
                     #[ 3,  4]], dtype=uint8)

type(k).__name__

l = np.array([[1,2],[3,4]])
m = np.array(([1,2],[3,4]))

n = np.arange(0,30,5)  # array([ 0,  5, 10, 15, 20, 25])
n = np.arange(1,30,5)  # array([ 1,  6, 11, 16, 21, 26])
n = np.arange(0,30,5).reshape(3,2) # array([[ 0,  5],
                                         # [10, 15],
                                         # [20, 25]])

n = np.arange(0,2)   # array([0, 1])
n = np.arange(0,2,0.25)   # array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75])
n.size # 8
n = np.linspace(0,2)  # array([ 0.        ,  0.04081633,  0.08163265, ...,  1.91836735, 1.95918367,  2.        ])
n.size  # 50
n = np.linspace(0,2,9) #array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
n.size  # 9
np.sin(45) # 0.80607549111591759
np.sin(np.rad2deg(45)) # 0.85090352453411844
np.sin(np.deg2rad(45)) # 0.70710678118654757

o = np.array(([1,2],[3,4]))
p = np.array(([4,3],[2,1]))

#  elementwise product
o * p  # array([[4, 6],
              # [6, 4]])
#  matrix product
o.dot(p)  # array([[ 8,  5],
                 # [20, 13]])
# another matrix product
np.dot(o,p)  # array([[ 8,  5],
                    # [20, 13]])

np.random((2,3))
np.random.random((2,3))

j = np.array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
k = j.reshape(6,2)
k.shape
mn = k.T
mn.shape
