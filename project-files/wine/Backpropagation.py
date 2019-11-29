import numpy as np

# input
x = np.array([[0.2351, 0.4016],
             [-0.1764, -0.1916],
             [0.3057, -0.9394],
             [0.5590, 0.6353],
             [-0.6600, -0.1175]])

# output
y = np.array([[1, 1, 0, 0, 0]]).T


def train():
    # initialize weights randomly
    # the nodes are 2 -> 5 -> 1
    np.random.seed(10)
    w0 = 2 * np.random.random((2, 5)) - 1
    w1 = 2 * np.random.random((5, 1)) - 1

    # iterations
    for i in range(50000):
        layer0 = x
        layer1 = sigmoid(np.dot(layer0, w0))
        layer2 = sigmoid(np.dot(layer1, w1))

        error = y - layer2
        layer2_delta = error * sigmoid(layer2, derivative=True)
        layer1_error = layer2_delta.dot(w1.T)
        layer1_delta = layer1_error * sigmoid(layer1, derivative=True)

        # update the weights
        w0 = w0 + layer0.T.dot(layer1_delta)
        w1 = w1 + layer1.T.dot(layer2_delta)

    # normalize the output to 1 and -1
    output_list = []
    for output in layer2:
        if output > 0.5:
            output_list.append(1)
        else:
            output_list.append(-1)

    # print out the results
    print('Now the mean error is: ')
    print(round(np.mean(abs(error)), 5))

    print('The output is: ')
    print(output_list)


# sigmoid function
def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


train()
