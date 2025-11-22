import numpy as np
import matplotlib.pyplot as plt
import math
import random as rand
from sklearn.metrics import ConfusionMatrixDisplay

# Zach, from Project 1
def generate_class(label, no_of_clusters, no_of_points_per_cluster):
    dataset = generate_dataset(no_of_clusters, no_of_points_per_cluster)

    dataset = shift_class(dataset)

    # Creates a matrix that is amount of samples in a class rows and 3 columns with the label
    labeled_dataset = np.full((no_of_clusters * no_of_points_per_cluster, 3), label)

    # Replaces first 2 columns with samples
    labeled_dataset[:, :2] = dataset
    return labeled_dataset

# Zach -Updated from project 2
def generate_dataset(no_of_clusters: int, no_of_points_per_cluster: int):
    random_data = np.random.randn(no_of_points_per_cluster, 2)

    mu_new, sigma_new, fi = generate_random_shifts()

    transformed_data = shift_data(random_data, mu_new, sigma_new, fi)

    data_matrix = transformed_data
    for i in range(no_of_clusters - 1):
        random_data = np.random.randn(no_of_points_per_cluster, 2)
        mu_new, sigma_new, fi = generate_random_shifts()

        transformed_data = shift_data(random_data, mu_new, sigma_new, fi)

        data_matrix = np.concatenate((data_matrix, transformed_data), axis=0)

    return data_matrix

# Zach, from Project 1
def generate_random_shifts():
    mu_upper = 5
    mu_lower = -5
    mu_new = np.array([rand.randint(mu_lower, mu_upper), rand.randint(mu_lower, mu_upper)])
    sigma_new = np.array([[rand.randrange(1, 4), 0], [0, rand.randrange(1, 4)]])
    fi = math.pi / (rand.randint(1, 4))

    return mu_new, sigma_new, fi

# Zach, from project 2
def generate_random_xy_shift():
    # Generates the shifts for the random shifted mean values
    upper_bound = 5
    lower_bound = -5
    x_mu = rand.randrange(lower_bound, upper_bound)
    y_mu = rand.randrange(lower_bound, upper_bound)

    return x_mu, y_mu

# Zach, from Project 1
def shift_data(data, mu, sigma, fi):
    cos_fi = math.cos(fi)
    sin_fi = math.sin(fi)
    rotate_matrix = np.array([[cos_fi, -sin_fi], [sin_fi, cos_fi]])

    Data_new = (data).dot(sigma)
    shifted_data = Data_new.dot(rotate_matrix.T) + mu

    return shifted_data

# Zach - Based on xy shifting of clusters from previous projects
def shift_class(dataset):
    mu_upper = 10
    mu_lower = -10
    mu = np.array([rand.randint(mu_lower, mu_upper), rand.randint(mu_lower, mu_upper)])

    shifted_class = dataset + mu

    return shifted_class

# Zach
# Generates two classes and plots them with different colors
# Splits dataset into training, cv, and testing and returns the matrices
def generate_split_dataset(num_clusters, num_samples, num_classes):
    class1 = generate_class(0, num_clusters, num_samples)

    class2 = generate_class(1, num_clusters, num_samples)

    plt.scatter(class1[:, 0], class1[:, 1], label = "Class 0")

    plt.scatter(class2[:, 0], class2[:, 1], label = "Class 1")

    plt.legend()
    
    plt.show()

    merged_dataset = np.concatenate((class1, class2), axis=0)

    num_of_total_samples = num_samples * num_clusters * num_classes

    # ChatGPT consulted on properly splitting dataset
    # Shuffles data so classes can be split randomly
    np.random.shuffle(merged_dataset)

    # Finds the bounds of where the splits occur
    first_split = round(0.6 * num_of_total_samples)
    second_split = round(0.8 * num_of_total_samples)

    # Splits matrix into training, cv, and test with found boundries
    training_set = merged_dataset[:first_split]
    cv_set = merged_dataset[first_split:second_split]
    test_set = merged_dataset[second_split:]

    return training_set, cv_set, test_set

# Zach - Updated from project 2
def EM_uwplatt_init(data_matrix, no_of_components):
    num_classes = 2
    num_clusters = no_of_components // num_classes

    # Calculates both the x_mean and the y_mean of the global data_matrix
    means = np.mean(data_matrix, axis=0)
    x_mean = means[0]
    y_mean = means[1]

    # Creates an array with size of no_of_component rows and 2 columns and then fill it with randomly offset x_mean and y_mean
    mean_matrix = np.zeros((no_of_components, 2))
    for i in range(no_of_components):
        x_mu, y_mu = generate_random_xy_shift()
        mean_matrix[i][0] = x_mean + x_mu
        mean_matrix[i][1] = y_mean + y_mu

    # Reshapes mean matrix to support the GMMs being in different rows
    mean_matrix = np.reshape(mean_matrix, (2, no_of_components))

    # Creates a matrix of no_of_component columns filled with 1 / no_of_components
    # Initializing component weights
    weights_matrix = np.full(no_of_components // 2, (1 / (no_of_components / 2)))
    # Had issues with concatenating a matrix with size (4,), so ChatGPT was consulted on how to modify (4,) matrix for concatenation
    row_weights = weights_matrix[np.newaxis, :]
    component_weights_matrix = np.concatenate((row_weights, row_weights), axis=0)

    # Calculates the global covariance matrix and saves it to cov_matrix
    glob_cov = np.cov(data_matrix[:, :2], rowvar=False)
    temp_cov_matrix = glob_cov

    # Rows - every other row (k2) indexes matrix of component K
    # Columns - every other column (m2) indexes the matrices of GMM m
    # Creates a matrix of shape (8, 4), which is essentially the 4 cov_matrix stacked in 2 columns for the 2 classes
    for _ in range(num_clusters - 1):
        temp_cov_matrix = np.concatenate((temp_cov_matrix, glob_cov), axis=0)
    cov_matrix = np.concatenate((temp_cov_matrix, temp_cov_matrix), axis=1)


    # Creates the extended matrix of no of sample rows and 2 + no_of_component + 1 columns, + 1 is for the labels
    n_samples, no_dim = data_matrix.shape
    # Does no_dim - 1 as the data matrix has an extra column for the labels
    no_dim = no_dim - 1
    ext_matrix = np.zeros((n_samples, no_dim + no_of_components + 1))
    ext_matrix[:, :no_dim] = data_matrix[:, :no_dim]
    ext_matrix[:, (no_dim + no_of_components):] = data_matrix[:, no_dim:]

    return ext_matrix, mean_matrix, cov_matrix, component_weights_matrix


# Brady
def EM_uwplatt_expectation(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix):
    ROWS = 0
    COLS = 1

    no_components = component_weights_matrix.shape[COLS]
    no_GMMS = mean_matrix.shape[ROWS]
    no_dim = cov_matrix.shape[COLS] // no_GMMS

    # Iterate Over Each GMM
    for m in range(0, no_GMMS):
        # Calculate pdfs
        ext_matrix_m = ext_matrix[(ext_matrix[:, -1] == m), :]
        pdfs_k = multivariate_gaussian(
            ext_matrix_m[:, :no_dim],
            mean_matrix[m, :no_dim],
            cov_matrix[0:no_dim, m * no_dim:(m * no_dim) + no_dim]
        )

        # Calculate likelihood of the sample in the GMM
        lik_samp_GMM = component_weights_matrix[m, 0] * pdfs_k

        # Calculate likelihood of observing samples
        for k in range(1, no_components):
            # Calculate pdfs for the kth component of the mth GMM
            pdfs_k = multivariate_gaussian(
                ext_matrix_m[:, :no_dim],
                mean_matrix[m, (no_dim * k):no_dim * (k + 1)], 
                cov_matrix[k * no_dim:(k * no_dim) + no_dim,
                m * no_dim:(m * no_dim) + no_dim]
            )

            # Calculate likelihood of the sample in the GMM
            lik_samp_GMM += component_weights_matrix[m, k] * pdfs_k

        for k in range(0, no_components):
            pdfs_k = multivariate_gaussian(
                ext_matrix_m[:, :no_dim],
                mean_matrix[m, (no_dim * k):no_dim * (k + 1)],
                cov_matrix[k * no_dim:(k * no_dim) + no_dim,
                m * no_dim:(m * no_dim) + no_dim]
            )

            # Calculate the likelihoods that the samples come from the k-th component for N-th model
            lik_samp_k = component_weights_matrix[m, k] * pdfs_k

            # Calculate & membership weights
            ext_matrix[(ext_matrix[:, -1] == m), no_dim + k + (m * (no_components - 1))] = np.divide(lik_samp_k, lik_samp_GMM)

        # Delete temp variables before moving to next GMM
        del lik_samp_k, lik_samp_GMM, pdfs_k, ext_matrix_m
    return (ext_matrix)


# Brady
def EM_uwplatt_maximization(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix):
    ROWS = 0
    COLS = 1

    no_components = component_weights_matrix.shape[COLS]
    no_GMMS = mean_matrix.shape[ROWS]
    no_dim = cov_matrix.shape[COLS] // no_GMMS

    # Iterate over each GMM
    for m in range(0, no_GMMS):
        ext_matrix_m = ext_matrix[(ext_matrix[:, -1] == m), :]
        # Iterate over each component of the GMM
        for k in range(0, no_components):
            # Calculate membership weights
            membership_weights = ext_matrix_m[:, no_dim + k + (m * (no_components - 1))]

            # Sum membership weights
            sum_weights = np.sum(membership_weights)

            # Get thew weighted samples
            weighted_samples = membership_weights.reshape(ext_matrix_m.shape[ROWS], 1) * ext_matrix_m[:, :no_dim]

            # Calculate new mean values and update mean matrix
            mean_matrix[m, (no_dim * k):no_dim * (k + 1)] = np.sum(weighted_samples, axis=0) / sum_weights

            # Get the differnce of each sample from the mean
            diff = ext_matrix_m[:, :no_dim] - mean_matrix[m, (no_dim * k):no_dim * (k + 1)]

            # Weight the differences by the membership weights
            weighted_diff = diff * membership_weights[:, np.newaxis]

            # Take the dot product of the transpose of weighted_diff and diff, 
            # and then divide by sum_weights to calculate the new covariance matrix
            cov_k = np.dot(weighted_diff.T, diff) / sum_weights

            # Update covariance matrix
            cov_matrix[
                k * no_dim:(k * no_dim) + no_dim, m * no_dim:(m * no_dim) + no_dim] = cov_k  # Assign to stacked format

        # Delete temp variables before moving to next GMM
        del ext_matrix_m, membership_weights, sum_weights, weighted_samples, diff, weighted_diff, cov_k

    return (mean_matrix, cov_matrix, component_weights_matrix)


# Ashton, Zach, Brady
def EM_uwplatt_dataset_log_likelihood(ext_matrix_m, mean_matrix, cov_matrix, cw_matrix):
 
    ROWS = 0
    COLS = 1
    no_components = cw_matrix.size
    no_GMMs = mean_matrix.size
    no_dim = cov_matrix.shape[COLS]
    log_likelihood = 0
    
    # data set log likeliness loop

    # iterate over all samples
    for i in range(ext_matrix_m.shape[ROWS]):
        x_i = ext_matrix_m[i, :no_dim]
        total_pdf = 0
        # PDF calculation loop
        for k in range(0, no_components):
            # running sum of PDFs for the current component
            
            log_likelihood += np.log(total_pdf.sum())
        
    return log_likelihood

def sample_log_likelihood(x, mean_matrix, cov_matrix, cw_matrix):
    K = cw_matrix.size
    D = 2

    total_pdf = 0
    for k in range(K):
        mu = mean_matrix[k * 2:k * 2 + 2]
        sigma = cov_matrix[k * 2:k * 2 + 2, :]
        w = cw_matrix[k]

        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt((2 * np.pi) ** D * sigma_det)
        fac = np.einsum('k,kl,l->', x - mu, sigma_inv, x - mu)
        pdf_val = np.exp(-fac / 2) / N

        total_pdf += w * pdf_val

    return np.log(total_pdf)

# Ashton
def classify_with_likelihood_ratio(test_set, gmm0, gmm1, threshold):
    mean0, cov0, w0 = gmm0
    mean1, cov1, w1 = gmm1

    y_true = test_set[:, -1].astype(int)
    X = test_set[:, :-1]
    y_pred = []

    for x in X:
        ll0 = sample_log_likelihood(x, mean0, cov0, w0)
        ll1 = sample_log_likelihood(x, mean1, cov1, w1)
        Lambda = ll1 - ll0
        y_pred.append(1 if Lambda > threshold else 0)
    y_pred = np.array(y_pred)

    return y_pred, y_true

# Brady, Ashton
def calc_confusion_and_accurracy(y_pred, y_true):
    # Confusion matrix new
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    true_neg = np.sum((y_pred == 0) & (y_true == 0))
    false_pos = np.sum((y_pred == 1) & (y_true == 0))
    false_neg = np.sum((y_pred == 0) & (y_true == 1))

    confusion_matrix = np.array([[true_pos, false_neg], [false_pos, true_neg]])
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    return confusion_matrix, accuracy

# Zach, Brady
# Based on previous converge testing, but updated to use accuracy
# Threshold started from 0.01 and shifted
def EM_test_accuracy_plateau(accuracy):
    # Iterate for a minimum number of iterations before checking accuracy
    if len(accuracy) < 12:
        return False
    else:
        # Threshold for determining when to stop
        threshold = 0.00001

        # Size of moving average
        avg_rng = 5
        diff = 0

        # Calculate moving average of the accuracy of the model
        for i in range(1, avg_rng - 1):
            diff += accuracy[-i] - accuracy[-(i + 1)]
        avgDiff = diff / avg_rng

        #print(f"Average Accuracy Difference: {avgDiff}")
        if abs(avgDiff) < threshold:
            return True
        else:
            return False

# From Canvas
def multivariate_gaussian(pos, mu, Sigma):
    # Return the multivariate Gaussian distribution on array pos.
    # pos is an array constructed by packing the meshed arrays of variables
    # x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    # Source: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    
    return np.exp(-fac / 2) / N

# Zach
# Splits the mean_matrix, cov_matrix, and cw_matrix for individual gmm
def split_matrix_into_gmm(mean_matrix, cov_matrix, component_weights_matrix):
    mean0 = mean_matrix[0, :]
    mean1 = mean_matrix[1, :]

    cov0 = cov_matrix[:, 0:2]
    cov1 = cov_matrix[:, 2:4]

    w0 = component_weights_matrix[0, :]
    w1 = component_weights_matrix[1, :]

    gmm0 = (mean0, cov0, w0)
    gmm1 = (mean1, cov1, w1)

    return gmm0, gmm1

# Zach, Brady
def EM_uwplatt_contour_plot(mean_matrix, cov_matrix, cw_matrix, label, iterations):
    no_components = cw_matrix.size
    no_GMMS = cw_matrix.shape[0]
    no_clusters = no_components // no_GMMS
    no_dim = cw_matrix.shape[0]

    # Calculate means for plot centering
    mean_x = mean_matrix[label, ::2].sum() // mean_matrix[label, ::2].size
    mean_y = mean_matrix[label, 1::2].sum() // mean_matrix[label, 1::2].size
    
    # Our 2-dimensional distribution will be over variables X and Y
    N = 40  # Number of ticks on X, Y axes
    X = np.linspace(mean_x - 10, mean_x + 10, N)
    Y = np.linspace(mean_y - 10, mean_y + 10, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))  # size (N, N, 2)
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Iterates through the number of components and calculate the multivariate gaussian and scales it to the component weight
    # Adds this to the z value that is used for the contour plot
    Z = 0
    for i in range(no_clusters):
        Z += cw_matrix[label, i] * multivariate_gaussian(
            pos,
            mean_matrix[label, (no_dim * i):no_dim * (i + 1)],
            cov_matrix[i * no_dim:(i + 1) * no_dim, 0:2]
        )

    # Adds the information to the contour plot and shows the plot
    plt.contourf(X, Y, Z)
    plt.title(f"Countour of GMM {label}; Iteration {iterations}")
    plt.show()

    return None  # Produces a contour plot

# Ashton, Titus
def det_curve(test_set, gmm0, gmm1):
    num_of_points = 50
    y_true = test_set[:, -1].astype(int)
    X = test_set[:, :-1]
    y_pred = []

    mean0, cov0, w0 = gmm0
    mean1, cov1, w1 = gmm1

    lambdas = []
    for x in X:
        ll0 = sample_log_likelihood(x, mean0, cov0, w0)
        ll1 = sample_log_likelihood(x, mean1, cov1, w1)
        lambdas.append(ll1 - ll0)
    lambdas = np.array(lambdas)

    thresholds = np.linspace(np.min(lambdas), np.max(lambdas), num_of_points)

    FARs = []
    FRRs = []

    err_found = False
    err = [[-1, -1], [-1, -1]]

    for thresh in thresholds:
        y_pred, y_true = classify_with_likelihood_ratio(test_set, gmm0, gmm1, thresh)
        cm, _ = calc_confusion_and_accurracy(y_pred, y_true)
        TP, FN = cm[0]
        FP, TN = cm[1]
        FAR = FP / (FP + TN)
        FRR = FN / (FN + TP)
        FARs.append(FAR)
        FRRs.append(FRR)
        if (FAR - FRR) < 0 and not err_found:
            print(f"threshold: {thresh}")
            err = cm
            err_found = True

    disp = ConfusionMatrixDisplay(confusion_matrix=err)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

    # all plt configuration within this function was made with a LLM
    plt.figure(figsize=(7, 6))
    plt.plot(FARs, FRRs, marker='o', linestyle='-', label='DET threshold curve')
    plt.plot((0,1), (0,1), linestyle='--', color='red', label='FAR = FRR (EER Line)')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title("DET Curve")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# Titus, from project 2, Edits from Brady
def EM_uwplatt(data_matrix, no_of_components, cv_matrix):
    completed = False

    ext_matrix, mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_init(data_matrix, no_of_components)

    dataset_log_likelihood_1 = []
    dataset_log_likelihood_2 = []
    accuracy_cv = []
    accuracy_train = []
    its = 0
    no_dim = 2

    while not completed:
        # Re-calculate extended matrix and upaate others
        ext_matrix = EM_uwplatt_expectation(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_maximization(
            ext_matrix,
            mean_matrix,
            cov_matrix,
            component_weights_matrix
        )

        # Plot the GMM contour
        EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix, 0, its)
        EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix, 1, its)
        
        # Track performance of GMM1
        likelihood1 = EM_uwplatt_dataset_log_likelihood(
            ext_matrix[(ext_matrix[:, -1] == 0), :],
            mean_matrix[0], cov_matrix[:, 0:no_dim], 
            component_weights_matrix[0]
        )
    
        dataset_log_likelihood_1.append(likelihood1)
    
        # Track performance of GMM2
        likelihood2 = EM_uwplatt_dataset_log_likelihood(
            ext_matrix[(ext_matrix[:, -1] == 1), :],
            mean_matrix[1], cov_matrix[:, no_dim:],
            component_weights_matrix[1]
        )
        
        dataset_log_likelihood_2.append(likelihood2)

        its += 1

        gmm0, gmm1 = split_matrix_into_gmm(mean_matrix, cov_matrix, component_weights_matrix)

        y_pred_train, y_true_train = classify_with_likelihood_ratio(np.delete(ext_matrix, slice(no_dim, -1), axis=1), gmm0, gmm1, 0)
        _, acc_train = calc_confusion_and_accurracy(y_pred_train, y_true_train)
        accuracy_train.append(acc_train)
        
        y_pred_cv, y_true_cv = classify_with_likelihood_ratio(cv_matrix, gmm0, gmm1, 0)
        cm, acc_cv = calc_confusion_and_accurracy(y_pred_cv, y_true_cv)
        accuracy_cv.append(acc_cv)
        print(f"acc_cv: {acc_cv}")

        completed = EM_test_accuracy_plateau(accuracy_cv)

    return mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_1, dataset_log_likelihood_2, accuracy_train, accuracy_cv, its


# Titus, Zach, from project 2
def main():
    num_clusters = 4
    num_classes = 2
    num_samples = 500
    # For dealing with a static dataset for testing
    # generate_static_dataset(num_components, 1000)
    # dataset = read_static_dataset(num_components)
    # mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_matrix, iterations = EM_uwplatt(dataset, num_components)

    training_set, cv_set, test_set = generate_split_dataset(num_clusters, num_samples, num_classes)

    # For testing the shifts of the dataset to try to get most of the generation to have ~20% class overlap
    # for _ in range(10):
    #     generate_split_dataset(num_clusters, num_samples, num_classes)


    # Runs the overall EM function
    mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_1, dataset_log_likelihood_2, accuracy_train, accuracy_cv, its = EM_uwplatt(
        training_set,
        num_clusters * num_classes, cv_set
    )

    gmm0, gmm1 = split_matrix_into_gmm(mean_matrix, cov_matrix, component_weights_matrix)

    det_curve(test_set, gmm0, gmm1)

    #print(f"iteration: {its}")
    #print(f"log_likelihood_1: {dataset_log_likelihood_1}")
    #print(f"log_likelihood_2: {dataset_log_likelihood_2}")
    
    x_log1 = list(range(len(dataset_log_likelihood_1)))
    y_log1 = dataset_log_likelihood_1

    x_log2 = list(range(len(dataset_log_likelihood_2)))
    y_log2 = dataset_log_likelihood_2

    x_acc_train = list(range(len(accuracy_train)))
    y_acc_train = accuracy_train
    
    x_acc_cv = list(range(len(accuracy_cv)))
    y_acc_cv = accuracy_cv

    # Graph parameters created using co-pilot
    plt.figure(figsize=(8, 5))
    plt.plot(x_log1, y_log1, marker="o", linestyle="-", color="black")
    plt.xlabel("Iterations for GMM 1")
    plt.ylabel("Log-Likelihood for GMM 1")
    plt.title("EM Convergence for GMM 1: Log-Likelihood Trend")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x_log2, y_log2, marker="o", linestyle="-", color="black")
    plt.xlabel("Iterations for GMM 2")
    plt.ylabel("Log-Likelihood for GMM 2")
    plt.title("EM Convergence for GMM 2: Log-Likelihood Trend")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x_acc_train, y_acc_train, marker="o", linestyle="-", color="black")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy for Training Set")
    plt.title("Accuracy of Training Set Vs. Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x_acc_cv, y_acc_cv, marker="o", linestyle="-", color="black")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy for C-V Set")
    plt.title("Accuracy of C-V Vs. Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()