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

from TestNeuralNetwork import *

# A neural network with:
INPUTS = 784
OUTPUTS = 10
SAMPLES_USED = 7000
#  - an output layer with 1 neuron (o1)


# Zach
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

# Zach
# Takes in the matrix and returns the mean of the matrix and the cov matrix of the matrix
def generate_mean_cov(matrix):
    mean_xy = np.mean(matrix, axis=0)
    cov = np.cov(matrix, rowvar=False)

    return mean_xy, cov

# Zach
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

def get_learning_rates() -> tuple[bool, list[float]]:
    valid = False;
    learning_rates = [float] * 3
    learning_rates[0] = float(input(
        f"Enter the starting value of your Learning Rate sweep: "
    ))
    if (learning_rates[0] < 0.0):
        print(f" {learning_rates[0]} is not a valid learning rate start value")
    else:
        learning_rates[1] = float(input(
            f"Enter the ending value of your Learning Rate sweep: "
        ))
        if (learning_rates[1] < 0.0 or learning_rates[1] < learning_rates[0]):
            print(f" {learning_rates[1]} is not a valid learning rate end value")
        else:
            learning_rates[2] = float(input(
                f"Enter step value of your Learning Rate sweep: "
            ))
            if (learning_rates[2] < 0.0 or
                    learning_rates[2] > learning_rates[1]):
                print(f" {learning_rates[2]} is not a valid learning rate step value")
            else:
                valid = True
    return valid, learning_rates

def get_hidden_neurons() -> tuple[bool, list[int]]:
    valid = False;
    hidden_neurons = [int] * 3
    hidden_neurons[0] = int(input(
        f"Enter the starting value of your Hidden Neuron sweep: "
    ))
    if (hidden_neurons[0] < 0):
        print(f" {hidden_neurons[0]} is not a valid hidden neuron start value")
    else:
        hidden_neurons[1] = int(input(
            f"Enter the ending step of your Hidden Neuron sweep: "
        ))
        if (hidden_neurons[1] < 0 or hidden_neurons[1] < hidden_neurons[0]):
            print(f" {hidden_neurons[1]} is not a valid hidden neuron end value")
        else:
            hidden_neurons[2] = float(input(
                f"Enter step value of your Hidden Neuron sweep: "
            ))
            if (hidden_neurons[2] < 0 or
                    hidden_neurons[2] > hidden_neurons[1]):
                print(f" {hidden_neurons[2]} is not a valid hidden neuron step value")
            else:
                valid = True
    return valid, hidden_neurons


def main():
    '''
    # Old code from skeleton code

    # df = generate_clusters()

    # print(df)

    # x = df.drop('target', axis=1)
    # y = df['target']

    # x = x.to_numpy()
    '''

    # Initilization loop
    run = False
    while not run:
        opcode = int(input(f"Which test would you like to run?\n"
                " 1. Normal Train & Plot\n"
                " 2. Learning Rate Sweep (Peak Learning Rate)\n"
                " 3. Hidden Layer Neurons Sweep (Peak Hidden Layer Neurons)\n"
                " 4. Four\n"
                " 5. Dual Sweep (Peak Balanced Accuracy)\n"
                "Please enter choose between 1, 2, 3, 4, or 5: "))
        if opcode < 1 or opcode > 5:
            print(f" {opcode} is not a valid option")
        else:
            network_set = False
            while not network_set:
                print("Please enter your network parameters as follows:")
                hidden_layers = int(input(
                    "Enter the number of hidden layers: "
                ))
                if hidden_layers < 1:
                    print(f" {hidden_layers} is not a valid number of hidden layers")
                else:
                    max_epochs = int(input(
                        "Enter the number the max number of epochs: "
                    ))
                    if max_epochs < 1:
                        print(f" {max_epochs} is not a valid number of max epochs")
                    else:
                        SAMPLES_USED = int(input(
                            "Enter the number of samples you would like to use: "
                        ))
                        network_set = True

            match opcode:
                case 1:
                    learning_set = False
                    while not learning_set:
                        learning_rates = float(input(
                            f"Enter the learning rate you would like: "
                        ))
                        if learning_rates < 0.0:
                            print(f" {learning_rates} is not a valid number of hidden neurons")
                        else:
                            learning_set = True

                    hidden_neurons_set = False
                    while not hidden_neurons_set:
                        hidden_neurons = int(input(
                            f"Enter the number of hidden neurons per layer you would like: "
                        ))
                        if hidden_neurons < 1:
                            print(f" {hidden_neurons} is not a valid number of hidden neurons")
                        else:
                            hidden_neurons_set = True
                    run = True
                case 2:
                    learning_set = False
                    while not learning_set:
                        learning_set, learning_rates = get_learning_rates()

                    hidden_neurons_set = False
                    while not hidden_neurons_set:
                        hidden_neurons = int(input(
                            f"Enter the number of hidden neurons per layer you would like: "
                        ))
                        if hidden_neurons < 1:
                            print(f" {hidden_neurons} is not a valid number of hidden neurons")
                        else:
                            hidden_neurons_set = True

                    run = True
                case 3:
                    hidden_neurons_set = False
                    while not hidden_neurons_set:
                        hidden_neurons_set, hidden_neurons = get_hidden_neurons()

                    learning_set = False
                    while not learning_set:
                        learning_rates = float(input(
                            f"Enter the learning rate you would like: "
                        ))
                        if learning_rates < 0.0:
                            print(f" {learning_rates} is not a valid number of hidden neurons")
                        else:
                            learning_set = True

                    run = True
                case 4:
                    hidden_neurons = 120
                    learning_rates = 0.01
                    run = True
                case 5:
                    learning_dual = False
                    hidden_neurons_dual = False
                    while not learning_dual:
                        learning_dual, learning_rates = get_learning_rates()
                    while not hidden_neurons_dual:
                        hidden_neurons_dual, hidden_neurons = get_hidden_neurons()
                    run = True


    # Grabbing dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)

    # Assign data -> x; Assign labels -> y;
    x, y = mnist['data'], mnist['target']

    # Converting training labels to integers.
    y = y.astype(np.int64)

    if opcode != 4:
        # normalize pixels to either black or white
        x = x / 255.0

    # Running less of the data so it runs faster
    x = x[:SAMPLES_USED, :]
    y = y[:SAMPLES_USED]

    # Instantiate MSE loss trend matrix
    mse_loss_trend = np.zeros((max_epochs))

    # Call test method based on opcode
    match opcode:
        case 1:
            # Create Poly Layer Neural Ntwork, train it, and plot it
            poly_layer_network = PolyLayerNetwork(hidden_layers, hidden_neurons,
                                                  learning_rates, max_epochs)
            train_and_plot_network(poly_layer_network, x, y)
        case 2:
            peak_learn_rate(x, y, hidden_layers,
                            hidden_neurons,learning_rates, max_epochs)
        case 3:
            peak_hidden_layer_neurons(x, y, hidden_layers,
                                      hidden_neurons,learning_rates, max_epochs)
        case 4:
            # No Longer works, RIP FOUR
            poly_layer_network = PolyLayerNetwork(hidden_layers, hidden_neurons,
                                                  learning_rates, max_epochs)
            train_and_plot_network(poly_layer_network, x, y)
        case 5:
            peak_balanced_accuracy(x, y, hidden_layers,
                                   hidden_neurons,learning_rates, max_epochs)


    # Functions that are used for testing for the best hyperparameters
    # peak_balanced_accuracy(x, y, learn_range, hidden_range_double)
    # peak_learn_rate(x, y, learn_range, 120)
    # peak_hidden_layers(x, y, learn_range[2], hidden_range_single)

# Code execution starts here
if __name__ == "__main__":
    main()
