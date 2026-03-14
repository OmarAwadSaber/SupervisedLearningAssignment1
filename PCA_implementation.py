from unittest import result

import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_        = None   # M  (Step 1)
        self.components_  = None   # Q  (Step 4)
        self.eigenvalues_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        n = X.shape[0]

        # Step 1: Compute the mean (M) 
        self.mean_ = np.mean(X, axis=0)

        # Step 2: Shift by mean  (F - M)
        shifted_mean = X - self.mean_

        # Step 3: Covariance matrix
        # Lecture divides by n (not n-1)
        cov = (shifted_mean.T @ shifted_mean) / n

        # Step 4: Eigenvalues & eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Step 5: Build Q — sort eigenvectors by |eigenvalue| DESC
        idx          = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues_ = eigenvalues[:self.n_components]

        # Q rows = eigenvectors (already normalized by eigh)
        self.components_  = eigenvectors[:, :self.n_components].T  # (k, n_features)

        self.explained_variance_ratio_ = (
            np.abs(self.eigenvalues_) / np.sum(np.abs(eigenvalues))
        )
        return self

    def transform(self, X):
        # Step 6: F` = Q (F - M)
        result = self.components_ @ (X - self.mean_).T
        return result.T.reshape(X.shape[0], self.n_components)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_projected):
        # Restore: F = (Q⁻¹ * F`) + M,  where Q⁻¹ = Qᵀ
        return (self.components_.T @ X_projected).T + self.mean_