import numpy as np
from sklearn.model_selection import train_test_split

# CHANGE TO PARAMETERS
INPUTS = 784
OUTPUTS = 10

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

# Supposedly the math for the derivative of softmax, I don't typically understand it.
#   Vastly improves performance though.
def softmax_jacobian_vec(p, g):
    # p: softmax output (10x1), g: dL/dp from MSE (10x1)
    # returns dL/dz (10x1)
    # J^T g = p * (g - <g, p>)
    dot = np.sum(g * p)          # scalar
    return p * (g - dot)         # elementwise

# A initialization method that attempts to generate weights for each layer with similar variance
#   to prevent heavy oscillation at higher learning rates
#   Perplexity link: https://www.perplexity.ai/search/can-you-explain-what-xavier-st-L7GkP3f.QZqlnlSrk6KSrA#0
def xavier(shape_in, shape_out):
    return np.random.normal(0, np.sqrt(1.0 / shape_in), size=(shape_out, shape_in))

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


class PolyLayerNetwork:
    def __init__(self, hidden_layers: int, hidden_layers_size: int, learning_rate: float, max_epoch: int) -> None:
        # Instantiate number of hidden layers
        self.hidden_layers: int = hidden_layers

        # Instantiate number of neurons per layer
        self.hidden_layers_size: int = hidden_layers_size

        # Instantiate learning rate
        self.learning_rate: float = learning_rate

        self.max_epoch: int = max_epoch

        # Nunber of weight and bias vectors needed is number_hidden_layers + 1
        #   But we will instantiate using lists for dynamic expansion for now
        #   (Should potentially make this static later by separating layer types)
        #   UPDATE: Should be updated now
        #       (i.e. input --> hidden, hidden-->hidden, hidden--> output)
        self.weights: list[np.ndarray] = [None] * (self.hidden_layers + 1)
        self.biases: list[np.ndarray] = [None] * (self.hidden_layers + 1)

        # Stage 1 (INPUTS --> HIDDEN LAYER 1) weights
        #   w[0] : (number_hidden_neurons, 784)
        #          (120, 784)
        self.weights[0] = xavier(INPUTS, self.hidden_layers_size)

        # Stage 1 (INPUTS --> HIDDEN LAYER) biases
        #   w[0] : (number_hidden_neurons, 1)
        #          (120, 1)
        self.biases[0] = np.random.normal(size=(self.hidden_layers_size, 1))

        for i in range(1, hidden_layers):
            # Stage i ( HIDDEN LAYER i --> HIDDEN LAYER(i + 1) ) weights
            #   w[i] : (number_hidden_neurons, number_hidden_neurons)
            #          (120, 120)
            self.weights[i] = xavier(self.hidden_layers_size, self.hidden_layers_size)

            # Stage i ( HIDDEN LAYER i --> HIDDEN LAYER(i + 1) ) biases
            #   w[i] : (number_hidden_neurons, 1)
            #          (120, 1)
            self.biases[i] = np.random.normal(size=(self.hidden_layers_size, 1))

        # Stage N (HIDDEN LAYER N --> OUTPUT) weights
        #   w[number_hidden_layers] : (10, number_hidden_neurons)
        #                             (10, 120)
        self.weights[self.hidden_layers] = xavier(self.hidden_layers_size, OUTPUTS)

        # Stage N (HIDDEN LAYER N --> OUTPUT) biases
        #   w[number_hidden_layers] : (10, 1)
        #                             (10, 1)
        self.biases[self.hidden_layers] = np.random.normal(size=(OUTPUTS, 1))

    def backProp(self, input: np.ndarray, sum_hidden: np.ndarray, act_hidden: np.ndarray,
                 predicted_labels: np.ndarray, ground_truth_labels: np.ndarray) -> None:
        # Sizing examples assume number_hidden_neurons = 120 & number_outputs = 10
        # x              : (input_size, 1) : (784, 1)
        # sum_hLayer[i]  : (hidden_layer_size, 1) for hidden layer i : (120, 1)
        # hLayer[i]      : (hidden_layer_size, 1) for hidden layer i : (120, 1)
        # y_pred, y_true : (output_size, 1) : (10, 1)

        x = input
        sum_hLayer, hLayer = sum_hidden, act_hidden
        y_pred, y_true = predicted_labels, ground_truth_labels

        ''' -- 1. dL/dy_pred -- '''
        # dL_dy_pred : (output_size, 1) :
        dL_dy_pred = (y_pred - y_true) # 2 * should be removed when running jacobian as it doubles effective LR

        ''' 
        -- 2. dL/dy_pred for Last Hidden Layer -- 
        # dL_sumLayer[-1] : (output_size, 1) : (10, 1)
        '''
        dL_dsumLayer = [None] * (self.hidden_layers + 1)

        # We start with the output layer --> last hidden layer
        #   We do so by calculating the partial derivative of the loss function with respect to the sum of the layer
        #   after it (in this case that's the output)
        #   We then multiply the partial derivative of the loss function with respect to the output activation (y_pred).
        #   This gives us the partial of the last hidden layer with respect to the softmax of the
        #   output layer
        #
        #   dL_dsum_out          : (output_size, 1) : (10, 1)

        dL_dsum_out = softmax_jacobian_vec(y_pred, dL_dy_pred)
        #dL_dsum_out = dL_dy_pred * softmax(sum_hLayer[-1])
        dL_dsumLayer[-1] = dL_dsum_out

        ''' 
        -- 3. backprop dL_dh through earlier hidden layers -- 
        # Reverse index weight & bias partials ("work backwards")
        # We'll start at the 2nd to last hidden layer since we already handled 
        #   Last Hidden Layer --> Output (dL_dsumh[-1])
        # self.weights[k+1].T  connects Hidden Layer k & Hidden Layer k + 1
        '''

        # Now we'll backprop through the rest of the hidden layers, if there are any, until we have the partial
        #   of the loss function with respect to the sigmoid activation of each layer
        #   If there's only one hidden layer, the for loop body is intentionally skipped
        for k in range(self.hidden_layers - 1, -1, -1):
            # tmp             : (hidden_layer_size, 1) : (120, 1)
            # dL_dsumLayer[k] : (hidden_layer_size, 1) : (120, 1)
            tmp = (self.weights[k+1].T).dot(dL_dsumLayer[k+1])
            dL_dsumLayer[k] = tmp * deriv_sigmoid(sum_hLayer[k])

        '''
       -- 4. Compute gradients for weights and biases --
       # First we have to instantiate our weight & bias grandient matrices
       #   dL_dwLayer[i] : (hidden_layer_size : hidden_layer_size) || (hidden_layer_size) 
       #                 : (120, 120), (120, 10)
       #   dL_dbLayer[i] : (hidden_layer_size : 1) || (output_size) : (120, 1) || (10, 1)
       '''
        dL_dwLayer = [None] * (self.hidden_layers + 1)
        dL_dbLayer = [None] * (self.hidden_layers + 1)

        # Now we can finally calculate our gradients.
        #   For dL_db, it's simply dL_dsumLayer.
        #   For dL_dw, we need to multiply by the activation of the previous layer
        #   (keep in mind we are moving forward now)
        for k in range(0, self.hidden_layers + 1):
            # Previous of first hidden layer is actually the inputs
            if k == 0:
                # x : (input_size, 1) : (784, 1)
                previous_activation = x
            else:
                # hLayer[k-1] : (hidden_layer_size : 1) : (120, 1)
                previous_activation = hLayer[k-1]

            dL_dbLayer[k] = dL_dsumLayer[k]
            dL_dwLayer[k] = dL_dsumLayer[k].dot(previous_activation.T)

            del previous_activation

        '''
        5. Update Weights & Biases with SGD Values
        # Finally we can update our weights and biases with our gradients and our learning rates
        #   weights[0]                    : (hidden_layer_size, input_size) : (120, 784)
        #   weights[1...hiden_layers - 1] : (hidden_layer_size, hidden_layer_size) : (120, 120)
        #   weights[hidden_layers]        : (output_size, hidden_layer_size) : (10, 120)
        '''
        for i in range(0, self.hidden_layers + 1):
            self.biases[i] -= self.learning_rate * dL_dbLayer[i]
            self.weights[i] -= self.learning_rate * dL_dwLayer[i]
        return None

    def feedforward(self, input: np.ndarray, labels: np.ndarray, train: bool) \
            -> np.ndarray:
        # x is a numpy array of 784 elements (e.g., flattened MNIST image).
        # -1 --> Collapse by one dimension
        x = input#.reshape(-1, 1)

        # z[0] : Input --> Hidden Layer 1
        # z[hidden_layers] : Hidden Layer n --> Output
        sum_hLayer: list[np.ndarray] = [None] * (self.hidden_layers + 1)

        # Size of h is 1 less than sum_h because last "h" value is used for o (out)
        hLayer: list[np.ndarray] = [None] * (self.hidden_layers)

        # Recall that range() has a non-inclusive upper-bound
        for i in range(0, self.hidden_layers + 1):
            '''
            # For Input --> Hidden Layer 1 : 
            # [hidden_neurons×784] @ [784×1] → [hidden_neurons×1]
            
            # For Hidden Layer i --> Hidden Layer i + 1:
            # [hidden_neurons×hidden_neurons] @ [hidden_neurons×1] → [hidden_neurons×1]
            
            # For Hidden Layer n --> Output:
            # [10×hidden_neurons] @ [hidden_neurons×1] → [10×1]
            '''

            # At beginning, use inputs (x), else use previous hidden neuron layer
            if i == 0:
                sum_hLayer[i] = self.weights[i].dot(x) + self.biases[i]
            else :
                sum_hLayer[i] = self.weights[i].dot(hLayer[i-1]) + self.biases[i]

            # Use sigmoid previous neuron pre-activation until output layer is reached
            if i < self.hidden_layers:
                hLayer[i] = sigmoid(sum_hLayer[i])
            else:
                o = softmax(sum_hLayer[i])

        if train:
            self.backProp(x, sum_hLayer, hLayer, o, labels)

        return o.flatten()

    def train(self, data, all_y_trues):
        # Split the training set into actual train and cross-validation sets
        data_train, data_cross_valid, y_train, y_cross_valid = train_test_split(
            data, all_y_trues, test_size=0.2,random_state=42)

        # Create an array to track MSE loss for training set
        mse_loss_trend_train = np.zeros(self.max_epoch)

        # Create an array to track MSE loss for training set
        mse_loss_trend_cross_validation = np.zeros(self.max_epoch)

        hidden_out = np.zeros

        # Epoch tracking
        epoch = 0
        stop = False
        while not stop and epoch < self.max_epoch:
            # Training loop with backprop
            for x, y_true in zip(data_train, y_train):
                # Perform a forward pass using feedforward()
                # -1 --> Collapse by one dimension

                input = x.reshape(-1, 1)
                labels = y_true.reshape(-1, 1)
                predicted_labels = self.feedforward(input, labels, train=True)

                # Assign softmax predictions to y_pred
                #y_pred = predicted_labels.reshape(-1, 1)

            # Feedforward pass on actual training set -> Generates data for calculating MSE Loss
            y_preds = []

            # This for loop replaces the previous syntax of:
            #   y = np.apply_along_axis(self.feedforward, 1, data)
            for x, y_true in zip(data_train, y_train):
                input = x.reshape(-1, 1)
                labels = y_true.reshape(-1, 1)
                y_preds.append(self.feedforward(input, labels, train=False))
            y_preds = np.array(y_preds)

            # Calculate and insert MSE Loss on training set for current epoch
            mse_loss_trend_train[epoch] = mse_loss(y_train, y_preds)

            # Feedforward pass on actual CV set -> Generates data for calculating MSE Loss

            # This for loop replaces the previous syntax of:
            #   y = np.apply_along_axis(self.feedforward, 1, data)
            y_preds_cross_valid = []
            for x, y_true in zip(data_cross_valid, y_cross_valid):
                input = x.reshape(-1, 1)
                labels = y_true.reshape(-1, 1)
                y_preds_cross_valid.append(self.feedforward(input, labels, train=False))
            y_preds_cross_valid = np.array(y_preds_cross_valid)

            # Calculate and insert MSE Loss on CV set for current epoch
            mse_loss_trend_cross_validation[epoch] = mse_loss(y_cross_valid, y_preds_cross_valid)

            # Changed because there was really no change after 10max_epochs
            # y_preds = np.apply_along_axis(self.feedforward, 1, data_train)
            loss = mse_loss(y_train, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))

            # Stop condition
            if epoch >= 12:
                # Moving window of 5, stop when avg diff is below vvvvvv
                stop = moving_avg_diff(mse_loss_trend_cross_validation, 5, epoch) < 0.0015
            epoch += 1

        return mse_loss_trend_train[:epoch], mse_loss_trend_cross_validation[:epoch]
    # Generates 2D data randomly that is shifted, rotated and scaled based on the based values
