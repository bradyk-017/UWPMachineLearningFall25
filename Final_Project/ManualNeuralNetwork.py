import numpy as np
from sklearn.model_selection import train_test_split

max_epoch = 100

# Gets a value and calculates the sigmoid activation function
# Returns this calculated value
def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

# Gets a value and calculates to the derivative of sigmoid
# Returns the calculated value
def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def softmax(z):
    exps = np.exp(z - np.max(z))  # stability trick
    return exps / np.sum(exps)

# This function takes in np arrays of the same length and calculates
# the square error for each sample and then finds the mean from those which is
# the mean squared error, which is then returned
def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

# Moving average difference function
# num_elements to use with np arrays: considers elements to be only num_elements long
def moving_avg_diff(elements, window_size, num_elements):
    if num_elements < 12:
        # Simply return average if less elements than the window size
        return sum(elements)/num_elements
    else:
        diff = 0
        # Calculate moving sum of differences
        for i in range(1, window_size):
            diff += abs(elements[num_elements - i] - elements[num_elements - i + 1])

        # Return average of 
        return diff / (window_size - 1)

class ManualNeuralNetwork:
    # This code is an edit of the other nueral network in the code.
    # It has been updated to use two hidden layers.
    def __init__(self, layers, learning_rate):
        # Instantiate number of hidden layers
        self.layers = layers

        # Instantiate learning rate
        self.learning_rate = learning_rate

        self.weights = []
        self.baises = []
        for i in range(len(self.layers) - 1):
            # Weights between each layer
            self.weights.append(np.random.normal(size=(self.layers[i + 1], self.layers[i])))
            # Baises of each layer (after input)
            self.baises.append(np.random.normal(size=(self.layers[i + 1])))

    def feedforward(self, x):
        # x is a 28x28 numpy input array with 784 elements
        # 28x28 to 784x1
        last_layer = x.flatten()

        # Layer by layer takes dot product (mult and sum) between current and next layer then adds baises
        for i in range(len(self.layers) - 1):
            if i > 0:
                last_layer = self.weights[i].dot(sigmoid(last_layer)) + self.baises[i]
            else:
                # Input layer, don't use activation function
                last_layer = self.weights[i].dot(last_layer) + self.baises[i]

        # Output layer, use softmax rather than sigmoid
        return softmax(last_layer).flatten()

    def train(self, data, all_y_trues):
        # Split the training set into actual train and cross-validation sets
        data_train, data_cross_valid, y_train, y_cross_valid = train_test_split(
            data, all_y_trues, test_size=0.2, random_state=42
        )

        mse_losses_training = np.zeros(max_epoch)
        mse_losses_cv = np.zeros(max_epoch)

        epoch = 0
        stop = False
        # Continue iterating while the moving average of MSE loss is
        # above the threshold and not hit max epochs
        while not stop and epoch < max_epoch:
            for x, y_true in zip(data_train, y_train):
                # x is a 28x28 numpy input array with 784 elements
                # 28x28 to 784x1

                h = [x.flatten()]
                # Layer by layer takes dot product (mult and sum) between current and next layer then adds baises
                for i in range(len(self.layers) - 1):
                    if i > 0:
                        h.append(self.weights[i].dot(sigmoid(h[-1])) + self.baises[i])
                    else:
                        # Input layer, don't use activation function
                        h.append(last_layer = self.weights[i].dot(h[-1]) + self.baises[i])

                # Output layer, use softmax rather than sigmoid
                y_pred = softmax(h[-1]).flatten()

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                # Stochastic Gradient Descent for output layer
                dL_dy = 2 * (y_pred - y_true)

                dL_dw2 = dL_dy.dot(h[1].T)
                dL_db2 = dL_dy
                dL_dh = self.weights2.T.dot(dL_dy)

                # Stochastic Gradient Descent for hidden layer
                dL_dz1 = dL_dh * deriv_sigmoid(h[1])
                dL_dw1 = dL_dz1.dot(h[0].T)
                dL_db1 = dL_dz1

                # --- Update weights and biases
                # Stage 2 (HIDDEN LAYER --> OUTPUT) weights
                self.weights2 -= self.learn_rate * dL_dw2
                self.bias2 -= self.learn_rate * dL_db2

                # Stage 1 (INPUTS --> HIDDEN LAYER) weights & biases
                self.weights1 -= self.learn_rate * dL_dw1
                self.bias1 -= self.learn_rate * dL_db1

                # # --- Calculate partial derivatives.
                # # --- Naming: dL_dw1 represents "partial L / partial w1"

                # # Output layer gradient
                # dL_dy = 2 * (y_pred - y_true)  # (10,1)
                # dL_dw3 = dL_dy.dot(h2.T)  # (output_size, hidden_size) = (10,4)
                # dL_db3 = dL_dy  # (10,1)
                # dL_dh2 = self.weights3.T.dot(dL_dy)  # (hidden_size,1) = (4,1)

                # # 2nd Hidden layer gradient
                # dL_dz2 = dL_dh2 * deriv_sigmoid(z2)
                # dL_dw2 = dL_dz2.dot(h1.T)
                # dL_db2 = dL_dz2
                # dL_dh1 = self.weights2.T.dot(dL_dz2)

                # # 1st Hidden layer gradient
                # dL_dz1 = dL_dh1 * deriv_sigmoid(z1)
                # dL_dw1 = dL_dz1.dot(x.T)
                # dL_db1 = dL_dz1

                # # --- Update weights and biases
                # # Stage 3 (HIDDEN LAYER 2 --> OUTPUT) weights & biases
                # self.weights3 -= self.learning_rate * dL_dw3
                # self.bias3 -= self.learning_rate * dL_db3

                # # Stage 2 (HIDDEN LAYER 1 --> HIDDEN LAYER 2) weights & biases
                # self.weights2 -= self.learning_rate * dL_dw2
                # self.bias2 -= self.learning_rate * dL_db2

                # # Stage 1 (INPUTS --> HIDDEN LAYER) weights & biases
                # self.weights1 -= self.learning_rate * dL_dw1
                # self.bias1 -= self.learning_rate * dL_db1

            # Feedforward pass on actual training set -> Generates data for calculating MSE Loss
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)

            # Calculate and insert MSE Loss on training set for current epoch
            mse_losses_training[epoch] = mse_loss(y_train, y_preds)

            # Feedforward pass on actual CV set -> Generates data for calculating MSE Loss
            y_preds_cross_valid = np.apply_along_axis(self.feedforward, 1, data_cross_valid)

            # Feedforward pass on actual CV set -> Generates data for calculating MSE Loss
            mse_losses_cv[epoch] = mse_loss(y_cross_valid, y_preds_cross_valid)

            # Changed because there was really no change after 10 epochs
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)
            loss = mse_loss(y_train, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))

            # Stop condition
            if epoch >= 12:
                # Moving window of 5, stop when avg diff is below vvvvvv
                stop = moving_avg_diff(mse_losses_cv, 5, epoch) < 0.0015

            epoch += 1
        return mse_losses_training[:epoch], mse_losses_cv[:epoch]