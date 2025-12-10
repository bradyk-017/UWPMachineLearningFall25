# Feedforward Neural Network with Backpropagation-Based Gradient Descent Training
# Slightly Modified Version of Original Code by Victor Zhou; https://victorzhou.com/blog/intro-to-neural-networks/
# Code Modifications by Hynek Boril for CS4030
# David's original code implements feedforward and backward passes with MSE loss
# Added by Hynek:
#    - Inference pass on the test set
#    - Hard Limit function applied on the top of the soft scores from the output layer
#    - Modified OurNeuralNetwork.train() to:
#          - Set aside a small cross-validation (CV) set
#          - Train network on the remaining part of the train set minus the CV set ('actual train set')
#          - Calculate and return MSE loss vectors for the actual training set and the CV set across training epochs
#    - Added generate_cluster() and generate_clusters() functions for synthetic dataset generation
#    - Added random data split to generate training and test sets from the synthetic dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import DetCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml
from statistics import mean
from IPython.display import display
import math
import pandas as pd
from sklearn.model_selection import train_test_split

# A neural network with:
INPUTS = 784
HIDDEN = 40
OUTPUTS = 10
SAMPLES_USED = 2100
    #  - an output layer with 1 neuron (o1)


learn_rate = 0.12
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
    exps = np.exp(z - np.max(z))   # stability trick
    return exps / np.sum(exps)


# This function takes in np arrays of the same length and calculates
# the square error for each sample and then finds the mean from those which is
# the mean squared error, which is then returned
def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    '''
    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''

    def __init__(self):


        # Weights
        self.weights1 = np.random.normal(size=(HIDDEN, INPUTS))
        self.weights2 = np.random.normal(size=(OUTPUTS, HIDDEN))

        # Biases
        self.bias1 = np.random.normal(size=(HIDDEN, 1))
        self.bias2 = np.random.normal(size=(OUTPUTS, 1))


    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        x = x.reshape(-1, 1)

        # Hidden layer: h = sigmoid
        z1 = self.weights1.dot(x) + self.bias1  # (4×784 @ 784×1) → (4×1)
        h = sigmoid(z1)

        # Output layer: z = W2*h + b2
        z2 = self.weights2.dot(h) + self.bias2  # (10×4 @ 4×1) → (10×1)
        o = softmax(z2)
        return o.flatten()

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        '''

        # Split the training set into actual train and cross-validation sets
        data_train, data_cross_valid, y_train, y_cross_valid = train_test_split(data, all_y_trues, test_size=0.2, random_state=42)

        mse_loss_trend_train = np.zeros((epochs))
        mse_loss_trend_cross_validation = np.zeros((epochs))
        epoch_counter = 0

        for epoch in range(epochs):
            for x, y_true in zip(data_train, y_train):

                x = x.reshape(-1, 1)
                y_true = y_true.reshape(-1, 1)

                # Hidden layer
                z1 = self.weights1.dot(x) + self.bias1
                h = sigmoid(z1)

                # Output layer
                sum_o1 = self.weights2.dot(h) + self.bias2
                o = softmax(sum_o1)
                y_pred = o
                y_pred = y_pred.reshape(-1, 1)

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
                # Output layer
                self.weights2 -= learn_rate * dL_dw2
                self.bias2 -= learn_rate * dL_db2

                # Hidden layer
                self.weights1 -= learn_rate * dL_dw1
                self.bias1 -= learn_rate * dL_db1

            # Feedforward pass on actual train set -> MSE loss on train
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)

            mse_loss_trend_train[epoch_counter] = mse_loss(y_train, y_preds)

            # Feedforward pass on CV set -> MSE loss on CV set
            y_preds_cross_valid = np.apply_along_axis(self.feedforward, 1, data_cross_valid)
            mse_loss_trend_cross_validation[epoch_counter] = mse_loss(y_cross_valid, y_preds_cross_valid)
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


class TwoLayerNeuralNetwork:
    '''
    This code is an edit of the other nueral network in the code. It has been updated to use
    two hidden layers.
    '''

    def __init__(self, hidden, learning_rate):

        self.hidden = hidden

        # Weights
        self.weights1 = np.random.normal(size=(self.hidden, INPUTS))
        self.weights2 = np.random.normal(size=(self.hidden, self.hidden))
        self.weights3 = np.random.normal(size=(OUTPUTS, self.hidden))

        # Biases
        self.bias1 = np.random.normal(size=(self.hidden, 1))
        self.bias2 = np.random.normal(size=(self.hidden, 1))
        self.bias3 = np.random.normal(size=(OUTPUTS, 1))

        self.learning_rate = learning_rate



    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        x = x.reshape(-1, 1)

        # Hidden layer: h = sigmoid
        z1 = self.weights1.dot(x) + self.bias1  # (4×784 @ 784×1) → (4×1)
        h1 = sigmoid(z1)

        z2 = self.weights2.dot(h1) + self.bias2
        h2 = sigmoid(z2)

        # Output layer: z = W2*h + b2
        z3 = self.weights3.dot(h2) + self.bias3  # (10×4 @ 4×1) → (10×1)
        o = softmax(z3)
        return o.flatten()

    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data.
        '''

        # Split the training set into actual train and cross-validation sets
        data_train, data_cross_valid, y_train, y_cross_valid = train_test_split(data, all_y_trues, test_size=0.2, random_state=42)

        mse_loss_trend_train = np.zeros((epochs))
        mse_loss_trend_cross_validation = np.zeros((epochs))
        epoch_counter = 0

        for epoch in range(epochs):
            for x, y_true in zip(data_train, y_train):
                # 1. Ensure column vectors
                x = x.reshape(-1, 1)  # (784,1)
                y_true = y_true.reshape(-1, 1)  # (10,1)

                # --- Do a feedforward (we'll need these values later)
                # Hidden layer: h = sigmoid
                z1 = self.weights1.dot(x) + self.bias1  # (4×784 @ 784×1) → (4×1)
                h1 = sigmoid(z1)

                z2 = self.weights2.dot(h1) + self.bias2
                h2 = sigmoid(z2)

                # Output layer: z = W2*h + b2
                z3 = self.weights3.dot(h2) + self.bias3  # (10×4 @ 4×1) → (10×1)
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
                # Output layer
                self.weights3 -= self.learning_rate * dL_dw3
                self.bias3 -= self.learning_rate * dL_db3

                # 2nd Hidden layer
                self.weights2 -= self.learning_rate * dL_dw2
                self.bias2 -= self.learning_rate * dL_db2

                # 1st Hidden layer
                self.weights1 -= self.learning_rate * dL_dw1
                self.bias1 -= self.learning_rate * dL_db1

            # Feedforward pass on actual train set -> MSE loss on train
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)

            mse_loss_trend_train[epoch_counter] = mse_loss(y_train, y_preds)

            # Feedforward pass on CV set -> MSE loss on CV set
            y_preds_cross_valid = np.apply_along_axis(self.feedforward, 1, data_cross_valid)
            mse_loss_trend_cross_validation[epoch_counter] = mse_loss(y_cross_valid, y_preds_cross_valid)
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


# Displays the side by side of two matrices, made for before and after scaling
def display_sbs_plots(before_matrix, after_matrix):
    plt.subplot(2, 1, 1)
    plt.scatter(before_matrix[:, 0], before_matrix[:, 1])
    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.title('Before Scaling')

    plt.subplot(2, 1, 2)
    plt.scatter(after_matrix[:, 0], after_matrix[:, 1])
    plt.xlim(-5, 25)
    plt.ylim(-5, 25)
    plt.title('After Scaling')

    plt.tight_layout()
    plt.show()

    return

# Takes in the matrix and returns the mean of the matrix and the cov matrix of the matrix
def generate_mean_cov(matrix):
    mean_xy = np.mean(matrix, axis=0)
    cov = np.cov(matrix, rowvar=False)

    return mean_xy, cov

# Takes the before and after matrix and displays the dataset mean and dataset covariance matrix of each
def display_mean_cov(before_matrix, after_matrix):
    b_mean_xy, b_cov = generate_mean_cov(before_matrix)

    a_mean_xy, a_cov = generate_mean_cov(after_matrix)

    print("---------------------------------------------------")
    print("Dataset mean before scaling:")
    print(b_mean_xy)
    print()
    print("Dataset covariance matrix before scaling:")
    print(b_cov)
    print("---------------------------------------------------")
    print("Dataset mean after scaling:")
    print(a_mean_xy)
    print()
    print("Dataset covariance matrix after scaling:")
    print(a_cov)
    print("---------------------------------------------------")

    return

# one hot vector
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def train_and_plot_network(network, x_train, y_train, x_test, y_test):
    # Call training function on our neural network, giving training data and labels as inputs
    mse_loss_trend_train, mse_loss_trend_cross_validation = network.train(x_train, y_train)
    plt.plot(mse_loss_trend_train, color='b', label="MSE Loss - Training Set")
    plt.plot(mse_loss_trend_cross_validation, color='r', label="MSE Loss - CV Set")
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    #all_y_predicted_soft = np.zeros((y_test.size))
    all_y_predicted_soft = np.apply_along_axis(network.feedforward, 1, x_test)
    all_y_predicted_hard = np.argmax(all_y_predicted_soft, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    balanced_accuracy = balanced_accuracy_score(y_test_labels, all_y_predicted_hard)

    print("Test balanced accuracy: ", balanced_accuracy)

    cm = confusion_matrix(y_test_labels, all_y_predicted_hard)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

    plt.show()

    return balanced_accuracy

def peak_balanced_accuracy(x, y):
    max_balanced_accuracy = 0.0
    peak_learning_rate = 0
    peak_nuerons = 0

    iteration = 0
    # Testing Network for performance over learning rate and nuerons per layer
    for i in np.arange(0.10, 0.25, 0.01):
        for j in np.arange(10, 310, 20):
            print(f"Iteration {iteration}")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Changes training and test labels from a single column to one-hot vectors
            y_train = one_hot(y_train, 10)
            y_test = one_hot(y_test, 10)

            two_layer_network = TwoLayerNeuralNetwork(j, i)

            balanced_accuracy = train_and_plot_network(two_layer_network, x_train, y_train, x_test, y_test)

            if balanced_accuracy > max_balanced_accuracy:
                max_balanced_accuracy = balanced_accuracy
                peak_learning_rate = i
                peak_nuerons = j

            iteration += 1

    print(
        f"Max balanced accuracy is {max_balanced_accuracy} with a learning rate of {peak_learning_rate} and {peak_nuerons} nuerons.")

def peak_learn_rate(x, y):
    peak_learning_rate = 0

    balanced_accuracy_trend = []

    for i in np.arange(0.10, 0.25, 0.01):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Changes training and test labels from a single column to one-hot vectors
        y_train = one_hot(y_train, 10)
        y_test = one_hot(y_test, 10)


        two_layer_network = TwoLayerNeuralNetwork(HIDDEN, i)

        balanced_accuracy = train_and_plot_network(two_layer_network, x_train, y_train, x_test, y_test)

        balanced_accuracy_trend.append(balanced_accuracy)

    plt.plot(balanced_accuracy_trend, color='b', label="Balanced Accuracy")
    plt.xlim(0, 15)
    plt.xlabel('Learning rate, 0.01 times x-value above 0.1')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    plt.show()




def main():
    #df = generate_clusters()

    #print(df)

    #x = df.drop('target', axis=1)
    #y = df['target']

    #x = x.to_numpy()
    
    # Grabbing dataset and formatting
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # Assign data -> x; Assign labels -> y;
    x, y = mnist['data'], mnist['target'] 

    # Converting training labels to integers.
    y = y.astype(np.int64)

    # Scaling down?
    x = x / 255.0

    #scaler = StandardScaler()
    #scaled_df = scaler.fit_transform(df)
    #scaled_df_no_labels = scaled_df[:, :2]

    #display_sbs_plots(x, scaled_df_no_labels)
    #display_mean_cov(x, scaled_df_no_labels)
    print(x.shape)
    # Running less of the data so it runs faster
    x = x[:SAMPLES_USED, :]
    y = y[:SAMPLES_USED]


    # Split the data into training data (x_train) and labels (y_train)
    # and a CV data (x_test) and labels (y_test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Changes training and test labels from a single column to one-hot vectors
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    # Instantiate MSE loss trend matrix
    mse_loss_trend = np.zeros((epochs))

    # Create Single Layer Neural Network and trains and plots it
    # one_layer_network = OurNeuralNetwork()
    # train_and_plot_network(one_layer_network, x_train, y_train, x_test, y_test)


    # Runs through iterations and returns best learning rate and nuerons
    # to gives the best balanced accuracy
    # peak_balanced_accuracy(x, y)

    # Runs through iterations to get the best learning rate
    # peak_learn_rate(x, y)


    two_layer_network1 = TwoLayerNeuralNetwork(130, 0.1)

    # two_layer_network2 = TwoLayerNeuralNetwork(70, 0.12)

    train_and_plot_network(two_layer_network1, x_train, y_train, x_test, y_test)

    # train_and_plot_network(two_layer_network2, x_train, y_train, x_test, y_test)



    # Code to try to get a 3D graph for looking for the best balanced accuracy
    # based on hidden nuerons and learning rate
    '''
    # Our 2-dimensional distribution will be over variables X and Y
    N = 15  # Number of ticks on X, Y axes
    X = np.linspace(0.1, 0.25, N)
    Y = np.linspace(10, 160, N)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros((15, 15))

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))  # size (N, N, 2)
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    X, Y = np.meshgrid(X, Y)


    # Adds the information to the contour plot and shows the plot
    plt.contourf(X, Y, Z)
    plt.title("Balanced Accuracy over learning rate and hidden layer size")
    plt.show()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Hidden Layer Neurons")
    ax.set_zlabel("Loss")

    plt.show()
    '''


# Code execution starts here
if __name__ == "__main__":
    main()
