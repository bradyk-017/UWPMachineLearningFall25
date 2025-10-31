import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import math
import random as rand
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import norm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Generates data from k-Means
# Zach, from Project 1
def generate_dataset(no_of_clusters: int, no_of_points_per_cluster: int):
    random_data = np.random.randn(no_of_points_per_cluster, 2)

    mu_new, sigma_new, fi = generate_random_shifts()

    transformed_data = shift_data(random_data, mu_new, sigma_new, fi)

    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    data_matrix = transformed_data
    for i in range(no_of_clusters - 1):
        random_data = np.random.randn(no_of_points_per_cluster, 2)
        mu_new, sigma_new, fi = generate_random_shifts()

        transformed_data = shift_data(random_data, mu_new, sigma_new, fi)

        plt.scatter(transformed_data[:, 0], transformed_data[:, 1])

        data_matrix = np.concatenate((data_matrix, transformed_data), axis=0)

    plt.show()

    return data_matrix

# Zach, from Project 1
def generate_random_shifts():
    mu_upper = 20
    mu_lower = -20
    mu_new = np.array([rand.randint(mu_lower, mu_upper), rand.randint(mu_lower, mu_upper)])
    sigma_new = np.array([[rand.randrange(1, 4), 0], [0, rand.randrange(1, 4)]])
    fi = math.pi / (rand.randint(1, 4))

    return mu_new, sigma_new, fi

# Zach
def generate_random_xy_shift():
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

# Zach
def EM_uwplatt_init(data_matrix, no_of_components):
    # Calculates both the x_mean and the y_mean of the global data_matrix
    means = np.mean(data_matrix, axis=0)
    x_mean = means[0]
    y_mean = means[1]

    # Creates an array with size of no_of_component rows and 2 columns and then fill it with radnomly offset x_mean and y_mean
    mean_matrix = np.zeros((no_of_components, 2))
    for i in range(no_of_components):
        x_mu, y_mu = generate_random_xy_shift()
        mean_matrix[i][0] = x_mean + x_mu
        mean_matrix[i][1] = y_mean + y_mu

    # Creates a matrix of no_of_component columns filled with 1 / no_of_components
    component_weights_matrix = np.full(no_of_components, (1 / no_of_components))

    # Calculates the global covariance matrix and saves it to cov_matrix
    glob_cov = np.cov(data_matrix, rowvar=False)
    cov_matrix = glob_cov

    # Creates a matrix of no_of_component global covariance matrices stacked
    for _ in range(no_of_components - 1):
        cov_matrix = np.concatenate((cov_matrix, glob_cov), axis=0)

    # Creates the extented matrix of no of sample rows and 2 + no_of_component columns
    n_samples, no_dim = data_matrix.shape
    ext_matrix = np.zeros((n_samples, no_dim + no_of_components))
    ext_matrix[:, :no_dim] = data_matrix

    return (ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)

# Brady
def EM_uwplatt_expectation(extended_matrix, mean_matrix, cov_matrix, component_weights_matrix):
    ROWS = 0
    COLS = 1

    no_samples = extended_matrix.shape[ROWS]
    # Used .siz instead of .shape as no_components only has 1 dimension when using a single GMM.
    no_components = component_weights_matrix.size
    no_dim = extended_matrix.shape[COLS] - no_components

    mean_k = np.sqrt(np.square(mean_matrix[0, 0]) + np.square(mean_matrix[0, 1]))
    std_dev_k = np.sqrt(np.square(cov_matrix[0, 0]) + np.square(cov_matrix[(0) + 1, 1]))

    # Create Gaus Component Obj
    gaus_comp = norm(loc=mean_k, scale=std_dev_k)

    pdfs_k = gaus_comp.pdf(extended_matrix[:, :no_dim - 1])

    # Calculate likelihood of the sample in the GMM
    lik_samp_GMM = component_weights_matrix[0] * pdfs_k

    # Calculate likelihood of observing samples
    for j in range(1, no_components):
        # Calculate the mean std_dev of the k-th component
        mean_k = np.sqrt(np.square(mean_matrix[j, 0]) + np.square(mean_matrix[j, 1]))
        std_dev_k = np.sqrt(np.square(cov_matrix[j * 2, 0]) + np.square(cov_matrix[(j * 2) + 1, 1]))

        # Create Gaus Component Obj
        gaus_comp = norm(loc=mean_k, scale=std_dev_k)

        # print(f"gaus_pdfs: {gaus_comp.pdf(extended_matrix[:, :no_dim-1])}")
        pdfs_k = gaus_comp.pdf(extended_matrix[:, :no_dim - 1])

        # Calculate likelihood of the sample in the GMM
        lik_samp_GMM += component_weights_matrix[j] * pdfs_k

    for k in range(0, no_components):
        # Create Gaus Component Obj
        gaus_comp = norm(loc=mean_k, scale=std_dev_k)

        pdfs_k = gaus_comp.pdf(extended_matrix[:, :no_dim - 1])

        # Calculate the likelihoods that the samples come from the k-th component for N-th model
        lik_samp_k = component_weights_matrix[k] * pdfs_k

        # Calculate & membership weights
        extended_matrix[:, no_dim + k:no_dim + k + 1] = np.divide(lik_samp_k, lik_samp_GMM)

    return (extended_matrix)

# Brady
def EM_uwplatt_maximization(extended_matrix, mean_matrix, cov_matrix, component_weights_matrix):
    ROWS = 0
    COLS = 1

    no_samples = extended_matrix.shape[ROWS]
    no_components = component_weights_matrix.size
    no_dim = extended_matrix.shape[COLS] - no_components

    # Calculate values for updates
    for k in range(0, no_components):
        # Membership weights of k-th component
        membership_weights = extended_matrix[:, no_dim + k]

        # Sum membership weights
        # .sum down the rows (column-wise)
        sum_membership_weights = np.sum(membership_weights, axis=ROWS)

        # Calculate & update component weight
        component_weights_matrix[k] = sum_membership_weights / no_samples

        # Multiply the samples by their membership weights
        weighted_samples = membership_weights.reshape(no_samples, 1) * extended_matrix[:, :no_dim]

        # Calculate the mean vector of the k-th component and update the mean matrix
        mean_matrix[k, :] = np.sum(weighted_samples, axis=ROWS) / sum_membership_weights

        # (w j,k for all samples) * [(x of all samples - mean x of all samples)^2]
        cov_num_xx = membership_weights * np.square(extended_matrix[:, 0] - mean_matrix[k, 0])

        # (w j,k for all samples) * (x of all samples - mean x of all samples) * (y of all samples - mean y of all samples)
        cov_num_xy = membership_weights * (extended_matrix[:, 0] - mean_matrix[k, 0]) * \
                     (extended_matrix[:, 1] - mean_matrix[k, 1])

        # (w j,k for all samples) * [(y of all samples - mean y of all samples)^2]
        cov_num_yy = membership_weights * np.square(extended_matrix[:, 0] - mean_matrix[k, 1])

        # Update cov xx for matrix of component k
        cov_matrix[(k * 2), 0] = np.sum(cov_num_xx, axis=ROWS) / sum_membership_weights

        # Update cov xy = cov yx for matrix of component k
        cov_matrix[(k * 2), 1] = np.sum(cov_num_xy, axis=ROWS) / sum_membership_weights
        cov_matrix[(k * 2) + 1, 0] = np.sum(cov_num_xy, axis=ROWS) / sum_membership_weights

        # Update cov yy for matrix of component k
        cov_matrix[(k * 2) + 1, 1] = np.sum(cov_num_yy, axis=ROWS) / sum_membership_weights

    return (mean_matrix, cov_matrix, component_weights_matrix)

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
def EM_uwplatt_contour_plot(mean_matrix, cov_matrix, cw_matrix):
    no_components = cw_matrix.size

    flat_max = np.max(mean_matrix, axis=0)
    flat_min = np.min(mean_matrix, axis=0)
    # Our 2-dimensional distribution will be over variables X and Y
    N = 60  # Number of ticks on X, Y axes
    # X = np.linspace(flat_min[0], flat_max[0], N)
    X = np.linspace(-35, 35, N)
    Y = np.linspace(-35, 35, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))  # size (N, N, 2)
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    Z = 0

    for i in range(no_components):
        Z += cw_matrix[i] * multivariate_gaussian(pos, mean_matrix[i], cov_matrix[i*2:i*2+2, :])

    # Create a surface plot and projected filled contour plot under it
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
    # Adjust the limits, ticks and view angle
    ax.set_zlim(0, 0.002)
    ax.set_zticks(np.linspace(0, 0.002, 5))
    ax.view_init(27, -21)
    plt.show()

    # Contour Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
    # Adjust the limits, ticks and view angle
    ax.set_zlim(0, 0.2)
    ax.set_zticks(np.linspace(0, 0.002, 5))
    ax.view_init(27, -21)
    plt.show()

    return None  # Produces a contour plot

# Ashton, Zach
def EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, cw_matrix):
    log_likelihood = 0
    K = len(cw_matrix)
    n = mean_matrix[0].shape[0]

    # data set log likeliness loop
    for i in range(ext_matrix.shape[0]):
        x_i = ext_matrix[i, :n]
        total_pdf = 0
        # PDF calculation loop
        for k in range(K):
            mu = mean_matrix[k]
            sigma = cov_matrix[k * 2: k * 2 + 2, :]
            w = cw_matrix[k]

            sigma_det = np.linalg.det(sigma)
            sigma_inv = np.linalg.inv(sigma)
            N = np.sqrt((2 * np.pi) ** n * sigma_det)
            fac = np.einsum('k,kl,l->', x_i - mu, sigma_inv, x_i - mu)
            total_pdf += w * np.exp(-fac / 2) / N
        # Added by Zach
        # total_pdf = total_pdf / K
        log_likelihood += np.log(total_pdf)

    # Added by Zach
    # log_likelihood = log_likelihood / ext_matrix.shape[0]
    return log_likelihood

# Ashton
def EM_uwplatt_test_convergence(log_likelihoods, iterations):
    # 1e-4 is supposedly the standard threshold for relative thresholding
    threshold = 1e-4
    if iterations < 1:
        return False
    previous = log_likelihoods[iterations - 1]
    current = log_likelihoods[iterations]
    # 1e-12 is added to prevent division by 0
    change = abs(current - previous) / (abs(previous) + 1e-12)
    return change < threshold

# Titus
def EM_uwplatt(data_matrix, no_of_components):
    has_completed = False

    ext_matrix, mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_init(data_matrix, no_of_components)

    dataset_log_likelihood_matrix = []
    iterations = 0

    while not has_completed and iterations < 10:
        # Re-calculate extended matrix and update others
        ext_matrix = EM_uwplatt_expectation(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_maximization(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        
        # Plot the GMM contour
        EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix)
        
        # Track perforcame of
        likelihood = EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        dataset_log_likelihood_matrix.append(likelihood)

        # Test the convergence to determine whether we should stop early
        # has_completed = EM_uwplatt_test_convergence(dataset_log_likelihood_matrix, iterations)
        iterations += 1
    return mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_matrix, iterations

# Titus, Zach
def main():
    num_components = 5
    mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_matrix, iterations = EM_uwplatt(
        generate_dataset(num_components, 1000), num_components)

    print(iterations)
    print(dataset_log_likelihood_matrix)

    x = list(range(len(dataset_log_likelihood_matrix)))
    y = dataset_log_likelihood_matrix

    # Graph parameters created using co-pilot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linestyle="-", color="black")
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")
    plt.title("EM Convergence: Log-Likelihood Trend")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
