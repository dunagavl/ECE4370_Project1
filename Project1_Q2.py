import numpy as np
import matplotlib.pyplot as plt
import json


# Enhanced PCA class with the missing method implemented
class pca:
    def __init__(self, d):
        # compute and reshape mean feature vector
        self.mn = np.mean(d, 0)[np.newaxis, :]
        # de-mean the features
        d_m = d - self.mn
        # compute covariance matrix
        covmat = d_m.T @ d_m / (len(d) - 1)
        # compute eigenvalues and eigenvectors
        evals, evects = np.linalg.eig(covmat)
        # sort evals and evects according to largest magnitude eigenvalues
        i = np.argsort(-np.abs(evals))
        self.evals = evals[i]
        self.evects = evects[:, i]

    def project(self, d):
        # de-mean the features and project the residuals onto the eigenvectors, resulting in coordinates
        # in the PCA feature space
        d_m = d - self.mn
        f = d_m @ self.evects
        return f

    def num_effective_dims(self, percvar_thresh):
        """
        Return N, the number of feature dimensions needed to represent
        at least percvar_thresh percent of the variance in the feature set
        """
        # Calculate total variance (sum of all eigenvalues)
        total_variance = np.sum(self.evals)

        # Calculate cumulative variance
        cumulative_variance = 0
        for i, eigenval in enumerate(self.evals):
            cumulative_variance += eigenval
            # Check if we've reached the threshold
            percent_explained = (cumulative_variance / total_variance) * 100
            if percent_explained >= percvar_thresh:
                return i + 1  # Return number of dimensions (1-indexed)

        # If we never reach the threshold, return all dimensions
        return len(self.evals)


def main():

    # Step 1: Load the data dictionary

    try:
        with open('../humact.json', 'r') as f:
            data_dict = json.load(f)

        # Extract the components we need
        features = np.array(data_dict['feat'])
        activity_ids = np.array(data_dict['actid'])
        activity_names = data_dict['actnames']

    except FileNotFoundError:
        print("Error: humact.json file not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Step 2: Perform PCA analysis
    p = pca(features)

    # Project data onto PCA space
    features_pca = p.project(features)

    # Display eigenvalue information
    print("Eigenvalues (variance in each PCA dimension):")
    total_var = np.sum(p.evals)
    for i in range(min(10, len(p.evals))):  # Show first 10
        percent = (p.evals[i] / total_var) * 100
        print(f"  PC{i + 1}: {p.evals[i]:.4f} ({percent:.1f}%)")

    # Calculate cumulative variance for first few components
    cumulative_var = np.cumsum(p.evals) / total_var * 100


    # Step 3: Create scatter plot with different colors/symbols for each activity
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    markers = ['o', '^', 's', 'D', '*']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each activity with unique color and marker
    unique_activities = np.unique(activity_ids)
    for i, act_id in enumerate(unique_activities):
        # Find samples belonging to this activity
        mask = activity_ids == act_id
        activity_data = features_pca[mask]


        ax.scatter(activity_data[:, 0], activity_data[:, 1],
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   label=activity_names[act_id - 1],  # activity IDs are 1-indexed
                   alpha=0.7, s=50)

    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.set_title('Human Activity Recognition: PCA Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Step 4: Compute correlation coefficients

    # Correlation between raw features 0 and 1
    rho_raw = np.corrcoef(features[:, 0], features[:, 1])[0, 1]
    print(f"Correlation between raw feature 0 and 1: ρ = {rho_raw:.6f}")

    # Correlation between PCA features 0 and 1
    rho_pca = np.corrcoef(features_pca[:, 0], features_pca[:, 1])[0, 1]
    print(f"Correlation between PCA feature 0 and 1: ρ = {rho_pca:.6f}")

    print(f"\nExplanation of PCA correlation:")
    print(f"The correlation between the first two PCA features is {rho_pca:.6f}")
    print("This value is expected because PCA finds orthogonal directions of maximum variance and "
          "orthogonal vectors are uncorrelated by definition")
    print()

    # Step 5: Use the num_effective_dims function

    # Test the function for 99.9% variance
    n_dims = p.num_effective_dims(99.9)
    print(f"Number of dimensions needed for 99.9% variance: {n_dims}")

if __name__ == "__main__":
    main()