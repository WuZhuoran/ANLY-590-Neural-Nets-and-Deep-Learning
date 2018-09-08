import random
import numpy as np
import pandas as pd
import timeit

# Generate a 10^5 list of random numbers
random_list = [random.randrange(0,1000000+1) for i in range(100000)]

# Generate a list of lists
random_list_of_lists = [[random.randrange(10) for _ in range(random.randrange(20))] for __ in range(random.randrange(30))]

# Append two lists together
l1 = [1,2,3]
l2 = [4,5]
l_t = l1 + l2

# Generate an 10^5 array

np_random_array = np.random.rand(100000)

# Using timeit add 1 to every element of the array using a loop and directly, compare the times

# Generate an array of arrays (3*2*2)
np_random_3d_array = np.random.rand(3,2,2)

# Generate a diagonal matrix
np_diag = np.diag([1,2,3,4,5])

# Using linalg find the inverse of [[1,2], [3,5]]
a = np.array([[1., 2.], [3., 4.]])
ainv = np.linalg.inv(a)

# Add two matrix together
m1 = np.matrix([[1, 2], [3, 4]])
m2 = np.matrix([[5, 6], [7, 8]])
m_t = m1 + m2

# Generate a 5*5 matrix of uniform random numbers using np.random
uniform_array = np.random.uniform(0,1,9)
uniform_matrix = np.reshape(uniform_array, (3,3))

# Transform it into a pandas dataframe
dt = pd.DataFrame(data=uniform_matrix[0:,0:])

# Using .iloc 