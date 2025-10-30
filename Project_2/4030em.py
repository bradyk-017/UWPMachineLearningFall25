import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import math
import random as rand
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import normal as gaus

# Generates data from k-Means
# Zach
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


# Zach
def generate_random_shifts():
    mu_new = np.array([rand.randint(-10, 10), rand.randint(-10, 10)])
    sigma_new = np.array([[rand.randrange(1, 4), 0], [0, rand.randrange(1, 4)]])
    fi = math.pi / (rand.randint(1, 4))

    return mu_new, sigma_new, fi

def generate_random_xy_shift():
    x_mu = rand.randrange(-100, 100) / rand.randint(10, 100)
    y_mu = rand.randrange(-100, 100) / rand.randint(10, 100)

    return x_mu, y_mu

# Zach
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
  component_weights_matrix = np.full(no_of_components, (1/no_of_components))

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
    gaus_comp = norm(loc = mean_k, scale = std_dev_k)

    pdfs_k = gaus_comp.pdf(extended_matrix[:, :no_dim-1])
        
    # Calculate likelihood of the sample in the GMM
    lik_samp_GMM = component_weights_matrix[0] * pdfs_k
    
    # Calculate likelihood of observing samples
    for j in range(1, no_components):
        #Calculate the mean std_dev of the k-th component
        mean_k = np.sqrt(np.square(mean_matrix[j, 0]) + np.square(mean_matrix[j, 1]))
        std_dev_k = np.sqrt(np.square(cov_matrix[j*2, 0]) + np.square(cov_matrix[(j*2) + 1, 1]))
        
        # Create Gaus Component Obj
        gaus_comp = norm(loc = mean_k, scale = std_dev_k)

        #print(f"gaus_pdfs: {gaus_comp.pdf(extended_matrix[:, :no_dim-1])}")
        pdfs_k = gaus_comp.pdf(extended_matrix[:, :no_dim-1])
        
        # Calculate likelihood of the sample in the GMM
        lik_samp_GMM += component_weights_matrix[j] * pdfs_k
    
    for k in range(0, no_components):
        # Create Gaus Component Obj
        gaus_comp = norm(loc = mean_k, scale = std_dev_k)

        pdfs_k = gaus_comp.pdf(extended_matrix[:, :no_dim-1])
        
        # Calculate the likelihoods that the samples come from the k-th component for N-th model
        lik_samp_k = component_weights_matrix[k] * pdfs_k
        
        # Calculate & membership weights
        extended_matrix[:, no_dim + k:no_dim +k+1] = np.divide(lik_samp_k, lik_samp_GMM)

    return (extended_matrix)

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
        sum_membership_weights = np.sum(membership_weights, axis = ROWS)
        
        # Calculate & update component weight
        component_weights_matrix[k] = sum_membership_weights / no_samples
        
        # Multiply the samples by their membership weights
        weighted_samples = membership_weights.reshape(no_samples, 1) * extended_matrix[:, :no_dim]

        # Calculate the mean vector of the k-th component and update the mean matrix
        mean_matrix[k, :] = np.sum(weighted_samples, axis = ROWS) / sum_membership_weights

        # (w j,k for all samples) * [(x of all samples - mean x of all samples)^2]
        cov_num_xx = membership_weights * np.square(extended_matrix[: , 0] - mean_matrix[k, 0])
        
        # (w j,k for all samples) * (x of all samples - mean x of all samples) * (y of all samples - mean y of all samples)
        cov_num_xy = membership_weights * (extended_matrix[: , 0] - mean_matrix[k, 0]) * \
            (extended_matrix[: , 1] - mean_matrix[k, 1]) 

        # (w j,k for all samples) * [(y of all samples - mean y of all samples)^2]
        cov_num_yy = membership_weights * np.square(extended_matrix[: , 0] - mean_matrix[k, 1])

        # Update cov xx for matrix of component k
        cov_matrix[(k * 2), 0] = np.sum(cov_num_xx, axis = ROWS) / sum_membership_weights

        # Update cov xy = cov yx for matrix of component k
        cov_matrix[(k * 2), 1] = np.sum(cov_num_xy, axis = ROWS) / sum_membership_weights
        cov_matrix[(k * 2) + 1, 0] = np.sum(cov_num_xy, axis = ROWS) / sum_membership_weights

         # Update cov yy for matrix of component k
        cov_matrix[(k * 2) + 1, 1] = np.sum(cov_num_yy, axis = ROWS) / sum_membership_weights
        
    return (mean_matrix, cov_matrix, component_weights_matrix)

def EM_uwplatt_contour_plot(mean_matrix, cov_matrix, cw_matrix):
  return None # Produces a contour plot


def EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, cw_matrix):
    log_likelihood = 0
    K = len(cw_matrix)
    n = mean_matrix[0].shape[0]

    for i in range(ext_matrix.shape[0]):
        x_i = ext_matrix[i]
        total_pdf = 0
        for k in range(K):
            mu = mean_matrix[k]
            sigma = cov_matrix[k * 2: k * 2 + 2, :]
            w = cw_matrix[k]

            sigma_det = np.linalg.det(sigma)
            sigma_inv = np.linalg.inv(sigma)
            N = np.sqrt((2 * np.pi) ** n * sigma_det)
            fac = np.einsum('k,kl,l->', x_i - mu, sigma_inv, x_i - mu)
            total_pdf += w * np.exp(-fac / 2) / N
        log_likelihood += np.log(total_pdf)

    return log_likelihood


def EM_uwplatt_test_convergence(dataset_log_likelihood_matrix, iterations):
        if iterations < 0:
            return False
        elif (dataset_log_likelihood_matrix[iterations] - dataset_log_likelihood_matrix[iterations - 1]) >= 0.01:
            return True
        else:
            return False


# e. During its runtime, EM_uwplatt() will collect and store dataset log likelihoods in a row
# matrix dataset_log_likelihood_matrix.
# f. The EM_uwplatt() function will return a reference to the mean_matrix,
# cov_matrix, component_weights_matrix, and
# dataset_log_likelihood_matrix together with the number of iterations it took for
# the algorithm to converge on the given dataset.
def EM_uwplatt(data_matrix, no_of_components):
    has_completed = False

    ext_matrix, mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_init(data_matrix, no_of_components)

    print(component_weights_matrix)
    dataset_log_likelihood_matrix = []
    iterations = 0

    while not has_completed:
        ext_matrix = EM_uwplatt_expectation(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_maximization(ext_matrix, mean_matrix, cov_matrix,
                                                                                    component_weights_matrix)
        EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix)
        likelihood = EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        dataset_log_likelihood_matrix.append(likelihood)
        has_completed = EM_uwplatt_test_convergence(dataset_log_likelihood_matrix, iterations)
        iterations += 1

    return mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_matrix, iterations


def main():
    num_components = 5
    mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_matrix, iterations = EM_uwplatt(
        generate_dataset(num_components, 1000), num_components)

    x = list(range(iterations))
    plt.plot(x, dataset_log_likelihood_matrix)
    plt.xlabel("Iterations")
    plt.ylabel("Likelihoods")
    plt.title("Log likelihoods trend")
    plt.show()

    # After the EM algorithm finishes, main() will produce a
    # 3D plot of the trained GMM pdf and its final contour plot.
    # It will also plot the dataset log likelihood trend over the training iterations.


if __name__ == "__main__":
    main()

