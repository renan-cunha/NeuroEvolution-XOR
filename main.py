from keras.models import Sequential
from sklearn.metrics import log_loss
from keras.layers import Dense
from typing import List, Tuple
import numpy as np
import keras
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random

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


def fitness(weights: np.ndarray) -> Tuple[float]:
    list_of_weights = pre_process_weights(weights)
    model.set_weights(list_of_weights)
    prob_one = model.predict(X)
    prob_zero = np.ones_like(prob_one) - prob_one
    probs = np.concatenate((prob_zero, prob_one), axis=1)
    return 1/log_loss(y, probs),


# Make the GA

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.gauss, 0, 1)
num_weights = get_num_weights(model)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
                 toolbox.attr_bool, n=num_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. It prevents
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5,6,7,8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2
    
    
toolbox.register("evaluate", fitness)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)


if __name__ == "__main__":
    random.seed(64)
    
    pop = toolbox.population(n=10)
    
    # Numpy equality function (operators.eq) between two arrays returns the
    # equality element wise, which raises an exception in the if similar()
    # check of the hall of fame. Using a different equality function like
    # numpy.array_equal or numpy.allclose solve this issue.
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.01, mutpb=1, ngen=4, stats=stats,
                        halloffame=hof)
    weights = pre_process_weights(hof[0])
    model.set_weights(weights)
    print("\nInput Bits")
    print(X)
    print("\nProb of being 1 | P(x) > 0.5 means that the output is 1")
    print(model.predict(X))
