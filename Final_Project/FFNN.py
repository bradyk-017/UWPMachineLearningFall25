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
from OneLayerNeuralNetwork import OurNeuralNetwork
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork

# A neural network with:
INPUTS = 784
HIDDEN = 40
OUTPUTS = 10
SAMPLES_USED = 2100
#  - an output layer with 1 neuron (o1)


learn_rate = 0.12
epochs = 100

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

    # Generate a MSE Loss line plot for the training set
    plt.plot(mse_loss_trend_train, color='b', label="MSE Loss - Training Set")

    # Generate a MSE line plot for the CV set
    plt.plot(mse_loss_trend_cross_validation, color='r', label="MSE Loss - CV Set")
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()

    # Create an array from the soft score predictions form the Neural Network
    all_y_predicted_soft = np.apply_along_axis(network.feedforward, 1, x_test)

    # Use argmax to create hard predictions from the soft predictions (probabilities)
    all_y_predicted_hard = np.argmax(all_y_predicted_soft, axis=1)

    # ?????
    y_test_labels = np.argmax(y_test, axis=1)

    # Pass ground-truth labels and predicted hard labels to calculate UAR
    balanced_accuracy = balanced_accuracy_score(y_test_labels, all_y_predicted_hard)

    # Print UAR
    print("Test balanced accuracy: ", balanced_accuracy)

    # Generate confusion matrix from ground-truth labels and hard predicted labels
    cm = confusion_matrix(y_test_labels, all_y_predicted_hard)

    # Display generated confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

    plt.show()

    return balanced_accuracy

def graph_balanced_accuracy(Z):
    peak_learning_rate = 0

    # Code to try to get a 3D graph for looking for the best balanced accuracy
    # based on hidden nuerons and learning rate
    # Our 2-dimensional distribution will be over variables X and Y
    N = 15  # Number of ticks on X, Y axes
    X = np.linspace(0, 15, N)
    Y = np.linspace(10, 310, N)
    X, Y = np.meshgrid(X, Y)



    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))  # size (N, N, 2)
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # X, Y = np.meshgrid(X, Y)


    # Adds the information to the contour plot and shows the plot
    plt.contourf(X, Y, Z)
    plt.title("Balanced Accuracy over learning rate and hidden layer size")
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlim(ax.get_xlim()[::-1])  # reverse X
    ax.set_ylim(ax.get_ylim()[::-1])  # reverse Y

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Hidden Layer Neurons")
    ax.set_zlabel("Balanced Accuracy")

    plt.show()


def graph_balanced_accuracy_heatmap(Z):
    peak_learning_rate = 0
    learning_rates = np.arange(0.10, 0.15, 0.01)        # 0.10–0.14 (5 values)
    neurons = (np.arange(5) * 20)                       # 0, 20, 40, 60, 80

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(Z, cmap='inferno', interpolation='bilinear', origin='lower')

    # Label ticks with real values
    ax.set_xticks(np.arange(len(neurons)))
    ax.set_yticks(np.arange(len(learning_rates)))

    ax.set_xticklabels(neurons)
    ax.set_yticklabels(learning_rates)

    ax.set_xlabel("Hidden Neurons")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Balanced Accuracy Heatmap")

    # Add numbers inside each cell
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", color="white")

    fig.colorbar(im, ax=ax, label="Balanced Accuracy")
    plt.show()



def peak_balanced_accuracy(x, y):
    max_balanced_accuracy = 0.0
    peak_learning_rate = 0
    peak_nuerons = 0

    Z = np.zeros((15, 15))

    iteration = 0
    # Testing Network for performance over learning rate and nuerons per layer
    for i in np.arange(0.05, 0.2, 0.01):
        for j in range(15):
            print(f"Iteration {iteration}")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Changes training and test labels from a single column to one-hot vectors
            y_train = one_hot(y_train, 10)
            y_test = one_hot(y_test, 10)

            two_layer_network = TwoLayerNeuralNetwork((j * 20) + 10, i)

            balanced_accuracy = train_and_plot_network(two_layer_network, x_train, y_train, x_test, y_test)


            Z[int(i * 100) - 5, j] = balanced_accuracy

            iteration += 1

    graph_balanced_accuracy(Z)

    graph_balanced_accuracy_heatmap(Z)

    print(
        f"Max balanced accuracy is {max_balanced_accuracy} with a learning rate of {peak_learning_rate} and {peak_nuerons} nuerons.")


def peak_learn_rate(x, y):
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
    '''
    # Old code from skeleton code

    # df = generate_clusters()

    # print(df)

    # x = df.drop('target', axis=1)
    # y = df['target']

    # x = x.to_numpy()
    '''

    # Grabbing dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # Assign data -> x; Assign labels -> y;
    x, y = mnist['data'], mnist['target']

    # Converting training labels to integers.
    y = y.astype(np.int64)

    # normalize pixels to either black or white
    x = x / 255.0

    '''
    # Old code from skeleton code
    
    # scaler = StandardScaler()
    # scaled_df = scaler.fit_transform(df)
    # scaled_df_no_labels = scaled_df[:, :2]

    # display_sbs_plots(x, scaled_df_no_labels)
    # display_mean_cov(x, scaled_df_no_labels)
    '''

    # print(x.shape) <-- Debugging? DELETE?


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
    peak_balanced_accuracy(x, y)

    # Runs through iterations to get the best learning rate
    # peak_learn_rate(x, y)

    # two_layer_network1 = TwoLayerNeuralNetwork(130, 0.1)
    #two_layer_network2 = TwoLayerNeuralNetwork(70, 0.12)

    # train_and_plot_network(two_layer_network1, x_train, y_train, x_test, y_test)

    #train_and_plot_network(two_layer_network2, x_train, y_train, x_test, y_test)

    # Code to try to get a 3D graph for looking for the best balanced accuracy
    # based on hidden nuerons and learning rate

    #train_and_plot_network(two_layer_network2, x_train, y_train, x_test, y_test)

# Code execution starts here
if __name__ == "__main__":
    main()
