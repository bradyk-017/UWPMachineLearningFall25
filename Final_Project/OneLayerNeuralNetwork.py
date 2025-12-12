import numpy as np
from sklearn.model_selection import train_test_split

# A neural network with:
INPUTS = 784
OUTPUTS = 10
SAMPLES_USED = 2100
#  - an output layer with 1 neuron (o1)

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

class OurNeuralNetwork:
    '''
    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''

    def __init__(self, hidden, learning_rate):

        # Instantiate number of hidden layers
        self.hidden = hidden

        # Instantiate learning rate
        self.learning_rate = learning_rate

        # Stage 1 (INPUTS --> HIDDEN LAYER) weights
        self.weights1 = np.random.normal(size=(self.hidden, INPUTS))

        # Stage 2 (HIDDEN LAYER --> OUTPUT) weights
        self.weights2 = np.random.normal(size=(OUTPUTS, self.hidden))

        # Stage 1 (INPUTS --> HIDDEN LAYER) biases
        self.bias1 = np.random.normal(size=(self.hidden, 1))

        # Stage 2 (HIDDEN LAYER --> OUTPUT) weights
        self.bias2 = np.random.normal(size=(OUTPUTS, 1))

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        # -1 --> Collapse by one dimension
        x = x.reshape(-1, 1)

        # Hidden layer: h (activation function) = sigmoid
        z1 = self.weights1.dot(x) + self.bias1  # [4×784] @ [784×1] → [4×1]
        h = sigmoid(z1)

        # Output layer: z = W2*h + b2
        z2 = self.weights2.dot(h) + self.bias2  # [10×4] @ [4×1] → [10×1]
        o = softmax(z2)
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
        mse_loss_trend_train = np.zeros(max_epoch)

        # Create an array to track MSE loss for training set
        mse_loss_trend_cross_validation = np.zeros(max_epoch)

        # Epoch tracking
        epoch = 0
        stop = False

        while not stop and epoch < max_epoch:
            for x, y_true in zip(data_train, y_train):
                # Perform one pass of feedfoward() pass manually as we will need some of the values later
                # -1 --> Collapse by one dimension
                x = x.reshape(-1, 1)
                y_true = y_true.reshape(-1, 1)

                # Hidden layer: h (activation function) = sigmoid
                z1 = self.weights1.dot(x) + self.bias1
                h = sigmoid(z1)

                # Output layer: z = W2*h + b2
                sum_o1 = self.weights2.dot(h) + self.bias2
                o = softmax(sum_o1)

                # Assign softmax preditions to y_pred
                y_pred = o.reshape(-1, 1)

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                # Stochastic Gradient Descent for output layer
                dL_dy = 2 * (y_pred - y_true)
                dL_dw2 = dL_dy.dot(h.T)
                dL_db2 = dL_dy
                dL_dh = self.weights2.T.dot(dL_dy)

                # Stochastic Gradient Descent for hidden layer
                dL_dz1 = dL_dh * deriv_sigmoid(z1)
                dL_dw1 = dL_dz1.dot(x.T)
                dL_db1 = dL_dz1

                # --- Update weights and biases
                # Stage 2 (HIDDEN LAYER --> OUTPUT) weights
                self.weights2 -= self.learning_rate * dL_dw2
                self.bias2 -= self.learning_rate * dL_db2

                # Stage 1 (INPUTS --> HIDDEN LAYER) weights & biases
                self.weights1 -= self.learning_rate * dL_dw1
                self.bias1 -= self.learning_rate * dL_db1

            # Feedforward pass on actual training set -> Generates data for calculating MSE Loss
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)

            # Calculate and insert MSE Loss on training set for current epoch
            mse_loss_trend_train[epoch] = mse_loss(y_train, y_preds)

            # Feedforward pass on actual CV set -> Generates data for calculating MSE Loss
            y_preds_cross_valid = np.apply_along_axis(self.feedforward, 1, data_cross_valid)

            # Calculate and insert MSE Loss on CV set for current epoch
            mse_loss_trend_cross_validation[epoch] = mse_loss(y_cross_valid, y_preds_cross_valid)

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

            # Stop condition
            if epoch >= 12:
                # Moving window of 5, stop when avg diff is below vvvvvv
                stop = moving_avg_diff(mse_loss_trend_cross_validation, 5, epoch) < 0.0015

            epoch += 1

        return mse_loss_trend_train, mse_loss_trend_cross_validation
    # Generates 2D data randomly that is shifted, rotated and scaled based on the based values
