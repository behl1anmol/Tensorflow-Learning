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


#Indexing
#Rules similar to Python
print('Single Axis Indexing')
x = tf.constant([1,2,3,4,5,6,7,8]) #rank 1 tensor
print(x.numpy())

print('From 2 before 7:', x[2:7].numpy())

#Multi Axis Indexing
print('Multi Axis Indexing')
print('RANK 2 TENSOR')
m_x = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) #rank 2 tensor
print(m_x.numpy())

print('single element:' ,m_x[2,2].numpy())
print('from row idx 1 to end and column idx 1 to end:',m_x[1:,1:].numpy())

print('RANK 3 TENSOR')
m3_x = tf.constant(      #rank 3 tensor shape (2,3,4)
    [
        [
            [1,2,3,100],
            [4,5,6,200],
            [12,14,15,300]
        ],
        [
            [7,8,9,400],
            [10,11,12,500],
            [16,17,18,600]
        ]
    ]
)
print(m3_x.numpy())
print('from batch idx 0 to end, from column idx 1 within a batch to end, starting from idx 2 features till end in every column:\n',m3_x[0:,1:,2:].numpy())

print('selecting last features accross all locations in each batch:\n', m3_x[:,:,-1].numpy())


#Reshaping a tensor

x = tf.constant([[1],[2],[3]])
print(x.shape)

print("Reshaping x to row vector:")
reshaped_x = tf.reshape(x,(1,3))
print(reshaped_x.shape)

print(x.numpy(),'\n',reshaped_x.numpy())

print("re-shaping on rank 3 vector:")
reshape_m3_x = tf.reshape(m3_x,[1,-1]) #passing -1 will flatten the tensor
print(reshape_m3_x.shape)
print(reshape_m3_x.numpy())

#reshaping from (2,3,4) to (3,2,4)
reshape_m3_x1 = tf.reshape(m3_x, (3,2,4))
print(reshape_m3_x1.shape)
print(reshape_m3_x1.numpy())

#reshaping from (2,3,4) to (12,-1) / (12,24)
reshape_m3_x1 = tf.reshape(m3_x, (3*4,-1))
print(reshape_m3_x1.shape)
print(reshape_m3_x1.numpy())

#transposing a tensor from (2,3,4) -> (4,3,2)
transpose_m3_x = tf.transpose(m3_x)
print(transpose_m3_x)


#Broadcasting in Tensorflow
print("Broadcasting")
a = tf.constant(2)
b = tf.constant([1,2,3])

print(tf.add(a,b)) #adds a to each element of b (Broadcasting)

reshaped_b = tf.reshape(b,[3,1])
c = tf.constant([1,2,3])
print(tf.multiply(reshaped_b,c)) #multiply each element of b with c and expands (Broadcasting)

print("broadcasting a vector:", tf.broadcast_to(tf.constant([1,2,3]),[3,3]))

#converting to tensor
arr = [1,2,3]
print("converting to tensor:", tf.convert_to_tensor(arr))

#Ragged Tensors
ragged_tensor = tf.ragged.constant([[1],[2,3],[4,5,6]])
print("ragged tensor:", ragged_tensor)
print("ragged tensor shape:", ragged_tensor.shape)

#string Tensors
string_tensor = tf.constant([b'hello world',b'welcome to tensorflow'])
print(string_tensor)
print("String to Number")
string_number_tensor = tf.constant('10')
print(tf.strings.to_number(string_number_tensor))

byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)

#sparse tensors
print("sparse tensors")
sparse_tensor = tf.sparse.SparseTensor(indices=[[1,0],[0,2]], values=[3,4], dense_shape=[3,3])
print(sparse_tensor)
print(tf.sparse.reorder(sparse_tensor)) #reorderes the sparse tensor as large indices supplied before smaller ones
print(tf.sparse.to_dense(tf.sparse.reorder(sparse_tensor)))













