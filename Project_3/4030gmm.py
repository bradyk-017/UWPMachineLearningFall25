import numpy as np
import matplotlib.pyplot as plt
import math
import random as rand
from scipy.stats import norm
from matplotlib import cm


# Generate static dataset for EM
def generate_static_dataset(no_of_clusters: int, no_of_points_per_cluster: int):
    dataset = generate_dataset(no_of_clusters, no_of_points_per_cluster)

    np.savetxt("static_dataset.csv", dataset, delimiter=",")
    return

# Gets the static dataset in the csv file for testing
def read_static_dataset(num_of_components):
    dataset = np.loadtxt("static_dataset.csv", delimiter=",")
    for i in range(num_of_components):
        offset = i * 1000
        plt.scatter(dataset[offset:(offset + 1000), 0], dataset[offset:(offset + 1000), 1])
    plt.show()
    return dataset

# Generates data from k-Means
# Zach, from Project 1

# TODO: modification for project 3:
# o Clusters within a class should be close to each other to form a tight group.
# o The overall inter-class overlap should be approximately 20%.
# o Assign class labels 0 and 1 to the two classes.
# Display a scatter plot of the dataset with distinct colors for each class, labeled axes, and a legend.

# a. Merge both class datasets and randomly split into:
# o Training set: 60%
# o Cross-validation (C-V) set: 20%
# o Test set: 20%
# b. Store both features and labels for each subset.
def generate_class(label, no_of_clusters, no_of_points_per_cluster  ):
    dataset = generate_dataset(no_of_clusters, no_of_points_per_cluster)

    dataset = shift_class(dataset)

    labeled_dataset = np.full((no_of_clusters * no_of_points_per_cluster, 3), label)

    labeled_dataset[:, :2] = dataset
    return labeled_dataset

def generate_dataset(no_of_clusters: int, no_of_points_per_cluster: int):
    random_data = np.random.randn(no_of_points_per_cluster, 2)

    mu_new, sigma_new, fi = generate_random_shifts()

    transformed_data = shift_data(random_data, mu_new, sigma_new, fi)

    # plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    data_matrix = transformed_data
    for i in range(no_of_clusters - 1):
        random_data = np.random.randn(no_of_points_per_cluster, 2)
        mu_new, sigma_new, fi = generate_random_shifts()

        transformed_data = shift_data(random_data, mu_new, sigma_new, fi)

        # plt.scatter(transformed_data[:, 0], transformed_data[:, 1])

        data_matrix = np.concatenate((data_matrix, transformed_data), axis=0)

    #plt.show()

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
    Data_new_rotated = Data_new.dot(rotate_matrix.T)
    shifted_data = Data_new.dot(rotate_matrix.T) + mu

    return shifted_data


def shift_class(dataset):
    mu_upper = 40
    mu_lower = -40
    mu = np.array([rand.randint(mu_lower, mu_upper), rand.randint(mu_lower, mu_upper)])

    shifted_class = dataset + mu

    return shifted_class

def generate_split_dataset(num_clusters, num_samples, num_classes):
    class1 = generate_class(0, num_clusters, num_samples)

    class2 = generate_class(1, num_clusters, num_samples)

    plt.scatter(class1[:, 0], class1[:, 1])

    plt.scatter(class2[:, 0], class2[:, 1])

    plt.show()

    merged_dataset = np.concatenate((class1, class2), axis=0)

    num_of_total_samples = num_samples * num_clusters * num_classes

    np.random.shuffle(merged_dataset)

    first_split = round(0.6 * num_of_total_samples)
    second_split = round(0.8 * num_of_total_samples)

    training_set = merged_dataset[:first_split]
    cv_set = merged_dataset[first_split:second_split]
    test_set = merged_dataset[second_split:]

    return training_set, cv_set, test_set


# Zach
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

    mean_matrix = np.reshape(mean_matrix, (2, no_of_components))

    #row_mean1 = mean_matrix[np.newaxis, :num_clusters]
    #row_mean2 = mean_matrix[np.newaxis, num_clusters:]
    #mean_matrix = np.concatenate((row_mean1, row_mean2), axis=0)

    # Creates a matrix of no_of_component columns filled with 1 / no_of_components
    # Initializing component weights
    weights_matrix = np.full(no_of_components // 2, (1 / (no_of_components/2)))
    row_weights = weights_matrix[np.newaxis, :]
    component_weights_matrix = np.concatenate((row_weights, row_weights), axis=0)

    # Calculates the global covariance matrix and saves it to cov_matrix
    glob_cov = np.cov(data_matrix[:, :2], rowvar=False)
    temp_cov_matrix = glob_cov

    # Rows - every other row (k2) indexes matrix of component K
    # Columns - every other column (m2) indexes the matrices of GMM m
    # Creates a matrix of no_of_component global covariance matrices stacked
    for _ in range(num_clusters - 1):
        temp_cov_matrix = np.concatenate((temp_cov_matrix, glob_cov), axis=0)
    cov_matrix = np.concatenate((temp_cov_matrix, temp_cov_matrix), axis=1)

    # Creates the extended matrix of no of sample rows and 2 + no_of_component columns
    n_samples, no_dim = data_matrix.shape
    no_dim = no_dim - 1
    ext_matrix = np.zeros((n_samples, no_dim + no_of_components + 1))
    ext_matrix[:, :no_dim] = data_matrix[:, :no_dim]
    ext_matrix[:, (no_dim + no_of_components):] = data_matrix[:, no_dim:]
    '''
    print("Mean Shape",mean_matrix.shape)
    print(mean_matrix)

    print("Cov Shape",cov_matrix.shape)
    print(cov_matrix)

    print("Component Matrix Shape",component_weights_matrix.shape)
    print(component_weights_matrix)

    print("Ext Shape", ext_matrix.shape)
    print(ext_matrix)
    '''
    
    return (ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)


# TO-DO
# Check that rework of expectation & maximization to iterate over each GMM works
# Verify that rework of indexing for multi-gmm data structures works
# Brady
def EM_uwplatt_expectation(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix):
    ROWS = 0
    COLS = 1

    no_samples = ext_matrix.shape[ROWS]
    no_components = component_weights_matrix.shape[COLS]
    no_GMMS = mean_matrix.shape[ROWS]
    no_dim = cov_matrix.shape[COLS] // no_GMMS

    # Iterate Over Each GMM
    for m in range(0, no_GMMS):
        # Calculate pdfs
        print(f"cov_matrix of GMM {m}: \n")
        for row in cov_matrix:
            print(' '.join(map(str, row)))
        ext_matrix_m = ext_matrix[(ext_matrix[:, -1] == m), :]
        pdfs_k = multivariate_gaussian(ext_matrix_m[:, :no_dim], 
                                       mean_matrix[m, :no_dim], cov_matrix[0:no_dim, m*no_dim:(m*no_dim)+no_dim])
    
        # Calculate likelihood of the sample in the GMM
        lik_samp_GMM = component_weights_matrix[m, 0] * pdfs_k
    
        # Calculate likelihood of observing samples
        for k in range(1, no_components):
            # Calculate pdfs for the kth component of the mth GMM
            pdfs_k = multivariate_gaussian(ext_matrix_m[:, :no_dim], mean_matrix[m, (no_dim*k):no_dim*(k+1)], 
                                           cov_matrix[k*no_dim:(k*no_dim)+no_dim, m*no_dim:(m*no_dim)+no_dim])
    
            # Calculate likelihood of the sample in the GMM
            lik_samp_GMM += component_weights_matrix[m, k] * pdfs_k
    
        for k in range(0, no_components):
            # Create Gaus Component Obj
            pdfs_k = multivariate_gaussian(ext_matrix_m[:, :no_dim], mean_matrix[m, (no_dim*k):no_dim*(k+1)], 
                                           cov_matrix[k*no_dim:(k*no_dim)+no_dim, m*no_dim:(m*no_dim)+no_dim])
    
            # Calculate the likelihoods that the samples come from the k-th component for N-th model
            lik_samp_k = component_weights_matrix[m, k] * pdfs_k
    
            # Calculate & membership weights
            ext_matrix[(ext_matrix[:, -1] == m), no_dim + k + (m * (no_components - 1))] = np.divide(lik_samp_k, lik_samp_GMM)#.reshape(ext_matrix_m.shape[ROWS],1)

    return (ext_matrix)

# Brady
def EM_uwplatt_maximization(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix):
    ROWS = 0
    COLS = 1

    no_samples = ext_matrix.shape[ROWS]
    no_components = component_weights_matrix.shape[COLS]
    no_GMMS = mean_matrix.shape[ROWS]
    no_dim = cov_matrix.shape[COLS] // no_GMMS

    # Iterate over each GMM
    for m in range(0, no_GMMS):
        ext_matrix_m = ext_matrix[(ext_matrix[:, -1] == m), :]
        # Iterate over each component of the GMM
        for k in range(0, no_components):
            # Calculate values for updates
            membership_weights = ext_matrix_m[:, no_dim + k + (m * (no_components - 1))]
            sum_weights = np.sum(membership_weights)
            weighted_samples = membership_weights.reshape(ext_matrix_m.shape[ROWS], 1) * ext_matrix_m[:, :no_dim]
            mean_matrix[m, (no_dim * k):no_dim * (k+1)] = np.sum(weighted_samples, axis=0) / sum_weights
    
            diff = ext_matrix_m[:, :no_dim] - mean_matrix[m, (no_dim * k):no_dim * (k+1)]
            weighted_diff = diff * membership_weights[:, np.newaxis]
            cov_k = np.dot(weighted_diff.T, diff) / sum_weights
    
            cov_matrix[k*no_dim:(k*no_dim)+no_dim, m*no_dim:(m*no_dim)+no_dim] = cov_k  # Assign to stacked format

    return (mean_matrix, cov_matrix, component_weights_matrix)


# Ashton, Zach
def EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, cw_matrix):
    log_likelihood = 0
    no_components = cw_matrix.size
    no_GMMS = cw_matrix.shape[0]
    no_clusters = no_components // no_GMMS
    no_dim = 2
    n = mean_matrix[0].shape[0]

    # data set log likeliness loop
    for i in range(ext_matrix.shape[0]):
        x_i = ext_matrix[i, :n]
        total_pdf = 0
        # PDF calculation loop
        for k in range(no_clusters):
            '''
            mu = mean_matrix[k]
            sigma = cov_matrix[k * 2:k * 2 + 2, k * 2:k *2 + 2]
            w = cw_matrix[k]

            sigma_det = np.linalg.det(sigma)
            sigma_inv = np.linalg.inv(sigma)
            N = np.sqrt((2 * np.pi) ** n * sigma_det)
            fac = np.einsum('k,kl,l->', x_i - mu, sigma_inv, x_i - mu)
            '''
            total_pdf += cw_matrix[0, k] * multivariate_gaussian(ext_matrix[i, :no_dim], mean_matrix[0, (no_dim*k):no_dim*(k+1)], cov_matrix[k*no_dim:(k+1)*no_dim, 0:2])
        log_likelihood += np.log(total_pdf)

    return log_likelihood

#ashton

def sample_log_likelihood(x, mean_matrix, cov_matrix, cw_matrix):
    K = cw_matrix.size
    D = 2
    no_dim = 2

    total_pdf = 0
    for k in range(K):

        mu = mean_matrix[k*2:k*2 + 2]
        sigma = cov_matrix[k*2:k*2 + 2, :]
        w = cw_matrix[k]

        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        N = np.sqrt((2 * np.pi) ** D * sigma_det)
        fac = np.einsum('k,kl,l->', x - mu, sigma_inv, x - mu)
        pdf_val = np.exp(-fac / 2) / N
        
        total_pdf += w * pdf_val

        # total_pdf += cw_matrix[0, k] * multivariate_gaussian(x, mean_matrix[0, (no_dim * k):no_dim * (k + 1)], cov_matrix[k * no_dim:(k + 1) * no_dim, 0:2])

    # total_pdf = max(total_pdf)
    return np.log(total_pdf)


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
    # Confusion matrix new
    true_pos = np.sum((y_pred == 1) & (y_true == 1))
    true_neg = np.sum((y_pred == 0) & (y_true == 0))
    false_pos = np.sum((y_pred == 1) & (y_true == 0))
    false_neg = np.sum((y_pred == 0) & (y_true == 1))

    confusion_matrix = np.array([[true_pos, false_neg],[false_pos, true_neg]])
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    return confusion_matrix, accuracy


def EM_uwplatt_test_convergence(log_likelihoods, iteration):
    min_iterations = 10
    threshold = 1e-4

    if iteration < min_iterations:
        return False

    diff = log_likelihoods[-1] - log_likelihoods[-2]
    if diff < threshold:
        return True
    else:
        return False

# From Canvas
def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    Source: https://scipython.com/blog/visualizing-the-bivariate-gaussian-
    distribution/
    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N

# Zach
def EM_uwplatt_contour_plot(mean_matrix, cov_matrix, cw_matrix, label):
    no_components = cw_matrix.size
    print("Components", no_components)
    no_GMMS = cw_matrix.shape[0]
    no_clusters = no_components // no_GMMS
    no_dim = 2
    print(cw_matrix.shape)
    print(no_clusters)


    # Our 2-dimensional distribution will be over variables X and Y
    N = 60  # Number of ticks on X, Y axes
    X = np.linspace(-30, 30, N)
    Y = np.linspace(-30, 30, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))  # size (N, N, 2)
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Iterates through the number of components and calculate the multivariate gaussian and scales it to the component weight
    # Adds this to the z value that is used for the contour plot
    Z = 0
    for i in range(no_clusters):
        Z += cw_matrix[label, i] * multivariate_gaussian(pos, mean_matrix[label, (no_dim*i):no_dim*(i+1)], cov_matrix[i*no_dim:(i+1)*no_dim, 0:2])

    # Adds the information to the contour plot and shows the plot
    plt.contourf(X, Y, Z)
    plt.show()

    return None  # Produces a contour plot

def EM_uwplatt_3D_plot(mean_matrix, cov_matrix, cw_matrix, label):
    no_components = cw_matrix.size
    no_GMMS = cw_matrix.shape[0]
    no_clusters = no_components // no_GMMS
    no_dim = 2

    # Our 2-dimensional distribution will be over variables X and Y
    N = 60  # Number of ticks on X, Y axes
    X = np.linspace(-40, 40, N)
    Y = np.linspace(-40, 40, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))  # size (N, N, 2)
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Iterates through the number of components and calculate the multivariate gaussian and scales it to the component weight
    # Adds this to the z value that is used for the 3D plot
    Z = 0
    for i in range(no_clusters):
        Z += cw_matrix[label, i] * multivariate_gaussian(pos, mean_matrix[label, (no_dim*i):no_dim*(i+1)], cov_matrix[i*no_dim:(i+1)*no_dim, 0:2])

    # 3D plotting from canvas
    # Create a surface plot and projected filled contour plot under it
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    # Adjust the limits, ticks and view angle
    ax.set_zlim(0, 0.02)
    ax.set_zticks(np.linspace(0, 0.02, 5))
    ax.view_init(27, -21)
    plt.show()

# TODO

# Ashton
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

    for thresh in thresholds:
        cm, _ = classify_with_likelihood_ratio(test_set, gmm0, gmm1, thresh)
        TP, FN = cm[0]
        FP, TN = cm[1]
        FAR = FP / (FP + TN)
        FRR = FN / (FN + TP)
        FARs.append(FAR)
        FRRs.append(FRR)



    #all plt configuration within this function was made with a LLM
    plt.figure(figsize=(7, 6))
    plt.plot(FARs, FRRs, marker='o', linestyle='-')
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title("DET Curve")
    plt.grid(True, which="both", ls="--")
    plt.show()


# Titus, from project 2, Edits from Brady
def EM_uwplatt(data_matrix, no_of_components, cv_matrix):
    completed_1 = False
    completed_2 = False
    completed_cv = False

    ext_matrix, mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_init(data_matrix, no_of_components)

    dataset_log_likelihood_1 = []
    dataset_log_likelihood_2 = []
    dataset_log_likelihood_cv = []
    its_1 = 0
    its_2 = 0
    #print(ext_matrix, mean_matrix, cov_matrix)

    while not completed_1 or not completed_2 or not completed_cv:
        # Re-calculate extended matrix and upaate others
        ext_matrix = EM_uwplatt_expectation(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_maximization(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
    
        # Plot the GMM contour
        EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix, 0)
        EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix, 1)
        # If GMM1 has not converged
        if not completed_1:
            # Track performance of GMM1
            likelihood1 = EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
            dataset_log_likelihood_1.append(likelihood1)
    
            # Test the convergence to determine whether we should stop GMM1
            completed_1 = EM_uwplatt_test_convergence(dataset_log_likelihood_1, its_1)
            its_1 += 1
        
        if not completed_2:
            # Track performance of GMM2
            likelihood2 = EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
            dataset_log_likelihood_2.append(likelihood2)
    
            # Test the convergence to determine whether we should stop GMM2
            completed_2 = EM_uwplatt_test_convergence(dataset_log_likelihood_2, its_2)
            its_2 += 1
        if not completed_cv:
            likelihood_cv = EM_uwplatt_dataset_log_likelihood(cv_matrix, mean_matrix, cov_matrix, component_weights_matrix)
            dataset_log_likelihood_cv.append(likelihood_cv)
            completed_cv = EM_uwplatt_test_convergence(dataset_log_likelihood_cv, its_1)

    return mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_1, dataset_log_likelihood_2, its_1, its_2


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

    # Runs the overall EM function
    mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_1, dataset_loglikelihood_2, its1, its2 = EM_uwplatt(training_set, num_clusters * num_classes, cv_set)
    print(mean_matrix.shape)
    print(cov_matrix.shape)

    # mean0, cov0, w0 = gmm0
    # mean1, cov1, w1 = gmm1
    mean0 = mean_matrix[0, :]
    mean1 = mean_matrix[1, :]

    cov0 = cov_matrix[:, 0:2]
    cov1 = cov_matrix[:, 2:4]

    w0 = component_weights_matrix[0, :]
    w1 = component_weights_matrix[1, :]


    gmm0 = (mean0, cov0, w0)
    gmm1 = (mean1, cov1, w1)

    det_curve(test_set, gmm0, gmm1)

    # print(iterations)
    # print(dataset_log_likelihood_matrix)

    # x = list(range(len(dataset_log_likelihood_matrix)))
    # y = dataset_log_likelihood_matrix

    # Shows the final 3D and contour plot
    # EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix)
    '''
    # Graph parameters created using co-pilot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linestyle="-", color="black")
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")
    plt.title("EM Convergence: Log-Likelihood Trend")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''

if __name__ == "__main__":
    main()
