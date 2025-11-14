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


# TODO

# Titus, from project 2
def EM_uwplatt(data_matrix, no_of_components):
    has_completed = False

    ext_matrix, mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_init(data_matrix, no_of_components)

    dataset_log_likelihood_matrix = []
    iterations = 0

    while not has_completed:
        # Re-calculate extended matrix and update others
        ext_matrix = EM_uwplatt_expectation(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        mean_matrix, cov_matrix, component_weights_matrix = EM_uwplatt_maximization(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)

        # Plot the GMM contour
        EM_uwplatt_contour_plot(mean_matrix, cov_matrix, component_weights_matrix)

        # Track performance of the algorithm
        likelihood = EM_uwplatt_dataset_log_likelihood(ext_matrix, mean_matrix, cov_matrix, component_weights_matrix)
        dataset_log_likelihood_matrix.append(likelihood)

        # Test the convergence to determine whether we should stop early
        has_completed = EM_uwplatt_test_convergence(dataset_log_likelihood_matrix, iterations)
        iterations += 1
    return mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_matrix, iterations

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

    print(training_set.shape)
    print(cv_set.shape)
    print(test_set.shape)

    # Runs the overall EM function
    # mean_matrix, cov_matrix, component_weights_matrix, dataset_log_likelihood_matrix, iterations = EM_uwplatt(generate_dataset(num_components, 500), num_components)

    # print(iterations)
    # print(dataset_log_likelihood_matrix)

    # x = list(range(len(dataset_log_likelihood_matrix)))
    # y = dataset_log_likelihood_matrix

    # Shows the final 3D and contour plot
    # EM_uwplatt_3D_plot(mean_matrix, cov_matrix, component_weights_matrix)
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
