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

POINTS_PER_CLUSTER = 400



learn_rate = 0.1
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


'''
This function takes in np arrays of the same length and calculates
the square error for each sample and then finds the mean from those which is
 the mean squared error, which is then returned
'''


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    '''
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)

    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''

    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements.
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

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
                # --- Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # Feedforward pass on actual train set -> MSE loss on train
            y_preds = np.apply_along_axis(self.feedforward, 1, data_train)
            mse_loss_trend_train[epoch_counter] = mse_loss(y_train, y_preds)

            # Feedforward pass on CV set -> MSE loss on CV set
            y_preds_cross_valid = np.apply_along_axis(self.feedforward, 1, data_cross_valid)
            mse_loss_trend_cross_validation[epoch_counter] = mse_loss(y_cross_valid, y_preds_cross_valid)
            epoch_counter += 1

            # --- Calculate total loss at the end of each 10 epochs
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data_train)
                loss = mse_loss(y_train, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

        return (mse_loss_trend_train, mse_loss_trend_cross_validation)

    # Generates 2D data randomly that is shifted, rotated and scaled based on the based values


# Returns the data
def generate_cluster(no_points, mean, sigmas, rotation_angle):
    Data = np.random.randn(no_points, 2)
    mu_new = mean
    sigma_new = sigmas
    fi = rotation_angle

    cos_fi = math.cos(fi)
    sin_fi = math.sin(fi)
    Rotate_Matrix = np.array([[cos_fi, -sin_fi], [sin_fi, cos_fi]])

    Data_new = (Data).dot(sigma_new)
    Data_new_rotated = Data_new.dot(Rotate_Matrix.T)
    Data_new_rotated_shifted = Data_new.dot(Rotate_Matrix.T) + mu_new

    return Data_new_rotated_shifted


# Generates the values to transform each cluster
# Calls the generate_cluster() function and passed the created values
# Plots the two clusters on the same scatterplot
# Combines the data into a single matrix and creates a corresponding matrix with the label of for each point
# based on the cluster
# Puts data into pandas dataframe with columns x, y, target
def generate_clusters():
    NO_POINTS_PER_CLASS = POINTS_PER_CLUSTER
    mean_1 = np.array([2, 2])
    mean_2 = np.array([10, 8])
    sigmas_1 = np.array([[2, 0], [0, 3]])
    sigmas_2 = np.array([[1.5, 0], [0, 5]])

    rotation_angle_1 = math.pi / 4
    rotation_angle_2 = -math.pi / 4
    no_points_1 = NO_POINTS_PER_CLASS
    no_points_2 = NO_POINTS_PER_CLASS

    Class_1 = generate_cluster(no_points_1, mean_1, sigmas_1, rotation_angle_1)
    Class_2 = generate_cluster(no_points_2, mean_2, sigmas_2, rotation_angle_2)

    plt.scatter(Class_1[:, 0], Class_1[:, 1], color='b')
    plt.scatter(Class_2[:, 0], Class_2[:, 1], color='r')

    axes = plt.gca()
    axes.set_aspect(aspect='equal')
    plt.show(block=False)
    plt.pause(2)  # show for 2 seconds (adjust as desired)
    plt.close()

    Classes_Pooled = np.concatenate((Class_1, Class_2), axis=0)

    target = np.concatenate((np.zeros((no_points_1)), np.ones((no_points_2))), axis=0)
    df = pd.DataFrame({'x': Classes_Pooled[:, 0], 'y': Classes_Pooled[:, 1], 'target': target})

    return df


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
#one hot vector
def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def main():
    #df = generate_clusters()

    #print(df)

    #X = df.drop('target', axis=1)
    #y = df['target']

    #X = X.to_numpy()
    # Grabbing dataset and formatting
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target']
    y = y.astype(np.int64)
    X = X / 255.0

    #scaler = StandardScaler()
    #scaled_df = scaler.fit_transform(df)
    #scaled_df_no_labels = scaled_df[:, :2]

    #display_sbs_plots(X, scaled_df_no_labels)
    #display_mean_cov(X, scaled_df_no_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)


    mse_loss_trend = np.zeros((epochs))
    network = OurNeuralNetwork()
    mse_loss_trend_train, mse_loss_trend_cross_validation = network.train(X_train, y_train)
    plt.plot(mse_loss_trend_train, color='b', label="MSE Loss - Training Set")
    plt.plot(mse_loss_trend_cross_validation, color='r', label="MSE Loss - CV Set")
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    #all_y_predicted_soft = np.zeros((y_test.size))
    all_y_predicted_soft = np.apply_along_axis(network.feedforward, 1, X_test)
    all_y_predicted_hard = np.argmax(all_y_predicted_soft, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("Test balanced accuracy: ", balanced_accuracy_score(y_test_labels, all_y_predicted_hard))

    cm = confusion_matrix(y_test_labels, all_y_predicted_hard)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

    plt.show()


# Code execution starts here
main()
