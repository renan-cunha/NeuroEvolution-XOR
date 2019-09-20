from keras.models import Sequential
from sklearn.metrics import log_loss
from keras.layers import Dense, Dropout
from typing import List, Tuple
import numpy as np
import keras

# Create neural network

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Handle pre processing of neural newtork weights

def get_shapes(model: keras.Model) -> List[Tuple[int]]:
    """Get a list with the shapes of a model matrices"""
    model_weights = model.get_weights()
    shapes = [x.shape for x in model_weights]
    return shapes


def get_num_weights(model: keras.Model) -> int:
    shapes = get_shapes(model)
    num_weights = [np.prod(x) for x in shapes]
    return sum(num_weights)


def pre_process_weights(new_weights: np.ndarray) -> List[np.ndarray]:
    """Takes a flatten cromossome of weights and make a lot of matrixes with
    it"""
    shapes = get_shapes(model)
    processed_weights: List[np.ndarray] = []

    i = 0
    for shape in shapes:
        num_weights = np.prod(shape)
        matrix = new_weights[i:i+num_weights]
        matrix = matrix.reshape(shape)
        processed_weights.append(matrix)
        i += num_weights

    return processed_weights


# Make fitness function

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])


def fitness(weights: np.ndarray) -> float:
    list_of_weights = pre_process_weights(weights)
    model.set_weights(list_of_weights)
    prob_one = model.predict(X)
    prob_zero = np.ones_like(prob_one) - prob_one
    probs = np.concatenate((prob_zero, prob_one), axis=1)
    return log_loss(y, probs)


num_weights = get_num_weights(model)
chromo = np.linspace(1, num_weights, num_weights)
print(fitness(chromo))
