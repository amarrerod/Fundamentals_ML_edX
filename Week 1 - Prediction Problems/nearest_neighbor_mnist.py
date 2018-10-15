
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree

# Definimos algunas funciones utiles
def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28, 28)), cmap = plt.cm.gray)
    plt.show()
    return

def visualize_image(index, dataset = "train"):
    if(dataset == "train"):
        show_digit(train_data[index])
        label = train_labels[index]
    else:
        show_digit(test_data[index])
        label = test_labels[index]
    print("Label: " + str(label))
    return


def squared_euclidean_distance(x, y):
    return np.sum(np.square(x - y))

def find_nearest_neighbor(x):
    distances = [squared_euclidean_distance(x, train_data[i, ]) for i in range(len(train_labels))]
    return np.argmin(distances)

def nearest_neighbor_classifier(x):
    index = find_nearest_neighbor(x)
    return train_labels[index]

train_data = np.load('NN_MNIST/MNIST/train_data.npy')
train_labels = np.load('NN_MNIST/MNIST/train_labels.npy')
test_data = np.load('NN_MNIST/MNIST/test_data.npy')
test_labels = np.load('NN_MNIST/MNIST/test_labels.npy')

print("Training dataset dimensions: ", np.shape(train_data))
print("Number of training labels: ", len(train_labels))
print("Test dataset dimension: ", np.shape(test_data))
print("Number of test labels: ", len(test_labels))
# Numero de ejemplos para cada digito de 0-9
train_digits, train_counts = np.unique(train_labels, return_counts = True)
print("Training set distribution: ")
print(dict(zip(train_digits, train_counts)))
test_digits, test_counts = np.unique(test_labels, return_counts = True)
print("Test set distribution: ")
print(dict(zip(test_digits, test_counts)))

# Tarea
x = test_data[100]
print("Indice del vecino mas cercano: ", find_nearest_neighbor(x))


# Full test set
t_before = time.time()
test_predictions = [nearest_neighbor_classifier(test_data[i,]) for i in range(len(test_labels))]
t_after = time.time()

## Compute the error
err_positions = np.not_equal(test_predictions, test_labels)
error = float(np.sum(err_positions))/len(test_labels)

print("Error of nearest neighbor classifier: ", error)
print("Classification time (seconds): ", t_after - t_before)

# Acelerando el algoritmo

t_before = time.time()
ball_tree = BallTree(train_data)
t_after = time.time()
t_training = t_after - t_before
print("Time to build data structure (seconds) using Ball Tree: ", t_training)
t_before = time.time()
test_neighbors = np.squeeze(ball_tree.query(test_data, k = 1, return_distance = False))
ball_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
t_testing = t_after - t_before
print("Time to classify test set (seconds): ", t_testing)
print("Does Ball tree produce the same predictions as above?:", np.array_equal(test_predictions, 
    ball_tree_predictions))

t_before = time.time()
kd_tree = KDTree(train_data)
t_after = time.time()
t_training = t_after - t_before
print("Time to build data structure (seconds) KDTree: ", t_training)
t_before = time.time()
test_neighbors = np.squeeze(kd_tree.query(test_data, k = 1, return_distance = False))
kd_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
t_testing = t_after - t_before
print("Time to classify test set (seconds): ", t_testing)
print("Does KD Tree produce the same predictions as above?", np.array_equal(test_predictions, kd_tree_predictions))