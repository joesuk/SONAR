import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

class PairRBFSampler(BaseEstimator, TransformerMixin):
    """
    Random Fourier Features for the RBF kernel using sin–cos *paired* features.

    Approximates: K(x, y) = exp(-gamma * ||x - y||^2)

    Mapping:
        For i=1..D (pairs), draw w_i ~ N(0, 2*gamma*I).
        phi_i(x) = sqrt(1/D) * [cos(x·w_i), sin(x·w_i)]
    Returns 2D features (cos & sin per frequency).

    Parameters
    ----------
    gamma : float
        RBF kernel parameter (same as sklearn's RBF: exp(-gamma * ||x - y||^2)).
    n_components : int
        Requested output dimensionality. If odd, the last 1 dim is dropped so the
        output has an even number (2 * n_pairs).
    random_state : int, RandomState instance or None
        Seed/control for reproducibility.
    """
    def __init__(self, gamma=1.0, n_components=100, random_state=None):
        self.gamma = float(gamma)
        self.n_components = int(n_components)
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)

        # Number of frequency draws (each yields a cos+sin pair)
        n_pairs = max(1, self.n_components // 2)   # ensure >=1
        self.n_features_in_ = X.shape[1]

        # w ~ N(0, 2*gamma I)
        scale = np.sqrt(2.0 * self.gamma)
        self.random_weights_ = rng.normal(
            loc=0.0, scale=scale, size=(self.n_features_in_, n_pairs)
        )

        # No random phase needed for paired map
        # Scaling so that E[phi(x)·phi(y)] ≈ K(x,y)
        self._scale_ = np.sqrt(1.0 / n_pairs)

        # Track actual output dimension (= 2*n_pairs)
        self.n_components_out_ = 2 * n_pairs
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["random_weights_", "_scale_", "n_components_out_"])
        X = np.asarray(X, dtype=float)

        # Project: Z = X @ W  -> shape (n_samples, n_pairs)
        Z = X @ self.random_weights_  # linear projections

        # Build paired features [cos(Z), sin(Z)] along last axis
        phi_cos = np.cos(Z)
        phi_sin = np.sin(Z)

        # Concatenate and scale
        Phi = np.concatenate([phi_cos, phi_sin], axis=1) * self._scale_

        # If user asked for odd n_components, we already rounded down to pairs.
        # Phi has shape (n_samples, 2*n_pairs).
        return Phi

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["n_components_out_"])
        return np.array([f"rff_pair_{i}" for i in range(self.n_components_out_)])


# return theoretically prescribed number of RFF's
def required_rff_features(D, lam, r_min=0.5, C=1.0):
    delta = lam / 2.0
    # ensure delta*r_min^2 < 1 to avoid log of <1
    if delta * r_min**2 >= 1:
        raise ValueError("delta * r_min^2 must be < 1 to get positive log argument")

    d = C * (D / (r_min**2)) * np.log(1.0 / (delta * r_min**2))
    return int(np.ceil(d))
