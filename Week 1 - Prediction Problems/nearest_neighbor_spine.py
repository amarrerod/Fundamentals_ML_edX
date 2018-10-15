
import numpy as np

def confusion(test_y, testy_L2):
    matrix = np.zeros([3, 3])
    for i in range(len(test_y)):
        true = int(test_y[i])
        predicted = int(testy_L2[i])
        matrix[true][predicted] += 1
    return matrix

def error_rate(test_y, test_fit):
    return float(sum(test_y != test_fit)) / len(test_y)

def distance_l2(x, y):
    return np.sum(np.square(x - y))

def distance_l1(x, y):
    return np.sum(np.abs(x - y))

def nearest_neighbor_l2(train_x, train_y, test_x):
    labels = np.ndarray(len(test_x))
    for i in range(len(test_x)):
        distances = [distance_l2(test_x[i], train_x[j]) for j in range(len(train_x))]
        labels[i] = train_y[np.argmin(distances)]
    return labels

def nearest_neighbor_l1(train_x, train_y, test_x):
    labels = np.ndarray(len(test_x))
    for i in range(len(test_x)):
        distances = [distance_l1(test_x[i], train_x[j]) for j in range(len(train_x))]
        labels[i] = train_y[np.argmin(distances)]
    return labels

labels = [b'NO', b'DH', b'SL']
data = np.loadtxt('NN_spine/column_3C.dat', converters = {6: lambda s:                                                                         labels.index(s)})

x = data[:, 0:6]
y = data[:, 6]
training_indices = list(range(0, 20)) + list(range(40, 188)) + list(range(230, 310))
test_indices = list(range(20, 40)) + list(range(188, 230))
train_x = x[training_indices, :]
train_y = y[training_indices] #Labels
test_x = x[test_indices, :]
test_y = y[test_indices] #Labels

# Usamos la distancia euclidea
test_l2 = nearest_neighbor_l2(train_x, train_y, test_x)
print(type(test_l2))
print(len(test_l2))
print(test_l2[40:50])

testy_L2 = nearest_neighbor_l2(train_x, train_y, test_x)
assert( type( testy_L2).__name__ == 'ndarray' )
assert( len(testy_L2) == 62 ) 
assert( np.all( testy_L2[50:60] == [ 0.,  0.,  0.,  0.,  2.,  0.,  2.,  0.,  0.,  0.]))
assert(np.all(testy_L2[0:10] == [ 0., 0., 0., 1., 1., 0., 1., 0., 0., 1.]))

# Probando la funcion nearest_neighbor l1
testy_L2 = nearest_neighbor_l2(train_x, train_y, test_x)
testy_L1 = nearest_neighbor_l1(train_x, train_y, test_x)

print(type( testy_L1) )
print(len(testy_L1) )
print(testy_L1[40:50] )
print(all(testy_L1 == testy_L2))

testy_L1 = nearest_neighbor_l1(train_x, train_y, test_x)
testy_L2 = nearest_neighbor_l2(train_x, train_y, test_x)

assert(type( testy_L1).__name__ == 'ndarray')
assert(len(testy_L1) == 62) 
assert(not all(testy_L1 == testy_L2))
assert(all(testy_L1[50:60] == [ 0., 2., 1., 0., 2., 0., 0., 0., 0., 0.]))
assert(all( testy_L1[0:10] == [ 0., 0., 0., 0., 1., 0., 1., 0., 0., 1.]))

print("Error rate of NN_L1: ", error_rate(test_y, testy_L1))
print("Error rate of NN_L2: ", error_rate(test_y, testy_L2))

# Probando el funcionamiento del metodo que calcula la matriz de confusion
L2_neo = confusion(test_y, testy_L2)  
print(type(L2_neo))
print(L2_neo.shape)
print(L2_neo)
L1_neo = confusion(test_y, testy_L1) 
print(L1_neo)
assert(type(L1_neo).__name__ == 'ndarray')
assert(L1_neo.shape == (3,3))
assert(np.all(L1_neo == [[ 16., 2., 2.],[ 10., 10., 0.],[ 0., 0., 22.]]))
L2_neo = confusion(test_y, testy_L2)  
assert(np.all(L2_neo == [[ 17., 1., 2.],[ 10., 10., 0.],[ 0., 0., 22.]]))