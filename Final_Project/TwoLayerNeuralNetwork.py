import numpy as np
from sklearn.model_selection import train_test_split


# A neural network with:
INPUTS = 784
OUTPUTS = 10
SAMPLES_USED = 2100
#  - an output layer with 1 neuron (o1)

epochs = 100


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



class TwoLayerNeuralNetwork:
    '''
    This code is an edit of the other nueral network in the code. It has been updated to use
    two hidden layers.
    '''

    def __init__(self, hidden, learning_rate):

        # Instantiate number of hidden layers
        self.hidden = hidden

        # Instantiate learning rate
        self.learning_rate = learning_rate

        # Stage 1 (INPUTS --> HIDDEN LAYER) weights
        self.weights1 = np.random.normal(size=(self.hidden, INPUTS))

        # Stage 2 (HIDDEN LAYER 1 --> HIDDEN LAYER 2) weights
        self.weights2 = np.random.normal(size=(self.hidden, self.hidden))

        # Stage 3 (HIDDEN LAYER 2 --> OUTPUT) weights
        self.weights3 = np.random.normal(size=(OUTPUTS, self.hidden))

        # Stage 1 (INPUTS --> HIDDEN LAYER) biases
        self.bias1 = np.random.normal(size=(self.hidden, 1))

        # Stage 2 (HIDDEN LAYER 1 --> HIDDEN LAYER 2) biases
        self.bias2 = np.random.normal(size=(self.hidden, 1))

        # Stage 3 (HIDDEN LAYER 2 --> OUTPUT) biases
        self.bias3 = np.random.normal(size=(OUTPUTS, 1))


    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        # -1 --> Collapse by one dimension
        x = x.reshape(-1, 1)

        # Hidden layer: h (activation function) = sigmoid
        z1 = self.weights1.dot(x) + self.bias1  # [4×784] @ [784×1] → [4×1]
        h1 = sigmoid(z1)

        # Output layer: z2 = W2*h1 + b2
        z2 = self.weights2.dot(h1) + self.bias2
        h2 = sigmoid(z2)

        # Output layer: z3 = W3*h2 + b3
        z3 = self.weights3.dot(h2) + self.bias3  # [10×4] @ [4×1] → [10×1]
        o = softmax(z3)
        return o.flatten()

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        '''

        # Split the training set into actual train and cross-validation sets
        data_train, data_cross_valid, y_train, y_cross_valid = train_test_split(data, all_y_trues, test_size=0.2,
                                                                                random_state=42)
        # Create an array to track MSE loss for training set
        mse_loss_trend_train = np.zeros((epochs))

        # Create an array to track MSE loss for training set
        mse_loss_trend_cross_validation = np.zeros((epochs))

        # Epoch tracking
        epoch_counter = 0

        # threshold for stopping condition
        threshold = .0001

        for epoch in range(epochs):
            for x, y_true in zip(data_train, y_train):
                # 1. Ensure column vectors
                x = x.reshape(-1, 1)  # (784,1)
                y_true = y_true.reshape(-1, 1)  # (10,1)

                # Perform one pass of feedfoward() pass manually as some of the values are needed later
                # Hidden layer: h = sigmoid
                z1 = self.weights1.dot(x) + self.bias1  # [4×784] @ [784×1] → [4×1]
                h1 = sigmoid(z1)

                z2 = self.weights2.dot(h1) + self.bias2
                h2 = sigmoid(z2)

                # Output layer: z = W2*h + b2
                z3 = self.weights3.dot(h2) + self.bias3  # [10×4] @ [4×1] → [10×1]
                o = softmax(z3)
                y_pred = o
                y_pred = y_pred.reshape(-1, 1)  # (10,1)
                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"

                # Output layer gradient
                dL_dy = 2 * (y_pred - y_true)  # (10,1)
                dL_dw3 = dL_dy.dot(h2.T)  # (output_size, hidden_size) = (10,4)
                dL_db3 = dL_dy  # (10,1)
                dL_dh2 = self.weights3.T.dot(dL_dy)  # (hidden_size,1) = (4,1)

                # 2nd Hidden layer gradient
                dL_dz2 = dL_dh2 * deriv_sigmoid(z2)
                dL_dw2 = dL_dz2.dot(h1.T)
                dL_db2 = dL_dz2
                dL_dh1 = self.weights2.T.dot(dL_dz2)

                # 1st Hidden layer gradient
                dL_dz1 = dL_dh1 * deriv_sigmoid(z1)
                dL_dw1 = dL_dz1.dot(x.T)
                dL_db1 = dL_dz1

                # --- Update weights and biases
                # Stage 3 (HIDDEN LAYER 2 --> OUTPUT) weights & biases
                self.weights3 -= self.learning_rate * dL_dw3
                self.bias3 -= self.learning_rate * dL_db3

                # Stage 2 (HIDDEN LAYER 1 --> HIDDEN LAYER 2) weights & biases
                self.weights2 -= self.learning_rate * dL_dw2
                self.bias2 -= self.learning_rate * dL_db2

                # Stage 1 (INPUTS --> HIDDEN LAYER) weights & biases
                self.weights1 -= self.learning_rate * dL_dw1
                self.bias1 -= self.learning_rate * dL_db1

            # Feedforward pass on actual training set -> Generates data for calculating MSE Loss
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)

            # Calculate and insert MSE Loss on training set for current epoch
            mse_loss_trend_train[epoch_counter] = mse_loss(y_train, y_preds)

            # Feedforward pass on actual CV set -> Generates data for calculating MSE Loss
            y_preds_cross_valid = np.apply_along_axis(self.feedforward, 1, data_cross_valid)

            # Calculate and insert MSE Loss on CV set for current epoch
            mse_loss_trend_cross_validation[epoch_counter] = mse_loss(y_cross_valid, y_preds_cross_valid)

            # Stopping condition that uses a threshold defined at the start of the function
            if np.abs(mse_loss_trend_cross_validation[epoch_counter] - mse_loss_trend_cross_validation[
                epoch_counter - 1]) < threshold:
                break
            epoch_counter += 1

            '''
            # --- Calculate total loss at the end of each 10 epochs
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data_train)
                loss = mse_loss(y_train, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))
            '''
            # Changed because there was really no change after 10 epochs
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)
            loss = mse_loss(y_train, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))

        return mse_loss_trend_train, mse_loss_trend_cross_validation
    # Generates 2D data randomly that is shifted, rotated and scaled based on the based values
