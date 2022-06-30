import tensorflow as tf
import numpy as np

print("#rank_0_tensor")
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

print("#rank_1_tensor")
rank_1_tensor = tf.constant([2.0,4.0,5.0])
print(rank_1_tensor)

print("#rank_2_tensor")
rank_2_tensor = tf.constant([[1,2],
                             [3,4],
                             [5,6],], dtype=tf.float16)
print(rank_2_tensor)

print("#rank_3_tensor")
#3 layers 2 rows per layer 5 columns in each layer -> shape = (3,2,5)
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]],

    [[10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19]],

    [[20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29]]
    ,])
print(rank_3_tensor)

rank_2_tensor_nparray = np.array(rank_2_tensor)
print(rank_2_tensor_nparray,'\n')

#Tensorflow Operations
print('#Tensorflow Operations')
a = tf.constant([[1,2],
                 [2,3]], dtype=tf.int32)
b = tf.ones([2,2],dtype=tf.int32)

print(tf.add(a,b),'\n') #element wise addition
print(tf.multiply(a,b),'\n') #element-wise multiplication
print(tf.matmul(a,b),'\n') #matrix multiplication

print(a+b,'\n') #element wise addition
print(a*b,'\n') #element-wise multiplication
print(a@b,'\n') #matrix multiplication


c = tf.constant([[4.0,5.0],[10.0,1.0]])

#Find the largest value
print(tf.reduce_max(c))
#Find the index of largest value
print(tf.argmax(c))
#Compute the softmax
print(tf.nn.softmax(c))


#learning about shapes
rank_4_tensor = tf.zeros([3,2,4,5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

#shape = (3,2,4,5)
#3 indicates number of sets
#every set has 4 rows(x axis), 2 columns(y axis), 5 set of row column along z axis




