"""Solution."""
import json

import numpy as np
from numpy import random
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
V_MEAN_PRIOR = 4  # specified in the task description


class BO_algo:
    def __init__(self):
        with open("config.json") as f:
            config = json.load(f)

        if config["f_kernel"] == "matern":
            f_kernel = Matern(nu=2.5)
        else:
            f_kernel = 0.5 * RBF(length_scale=config["f_length_scale"])
        self.f_hat = GaussianProcessRegressor(f_kernel, alpha=0.15)

        if config["v_kernel"] == "matern":
            v_kernel = Matern(nu=2.5)
        else:
            v_kernel = np.sqrt(2) * RBF(length_scale=config["v_length_scale"])
        self.v_hat = GaussianProcessRegressor(v_kernel, alpha=0.0001)

        self.xs = np.empty((0, 1))  # Initialize empty array for inputs
        self.fs = np.empty((0, 1))  # Initialize empty array for outputs
        self.vs = np.empty((0, 1))  # Initialize empty array for constraints

        self.is_hard = False
        self.default_to_safety = config["default_to_safety"]
        self.pertubation_step = config["pertubation_step"]

        self.v_confidence_level = config["v_confidence_level"]
        self.f_confidence_level = config["f_confidence_level"]
        self.f_confidence_level_decay = config["f_confidence_level_decay"]

        self.acquisition_function = config["acquisition_function"]
        self.reward_coef = config["reward_coef"]
        self.safety_coef = config["safety_coef"]
        self.safety_constraint_coef = config["safety_constraint_coef"]

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # if self.default_to_safety and self.is_hard:
        #     return self._best_safe_point()

        # Fit the GPs with the current data
        self.f_hat.fit(self.xs, self.fs)
        self.v_hat.fit(self.xs, self.vs - V_MEAN_PRIOR)

        candidate = self.optimize_acquisition_function()
        is_safe = self._v_upper(candidate) < SAFETY_THRESHOLD

        self.f_confidence_level -= self.f_confidence_level_decay
        self.f_confidence_level = max(self.f_confidence_level, 1e-3)

        if not is_safe and self.default_to_safety:
            delta = np.random.rand() * self.pertubation_step * np.random.choice([-1, 1])
            candidate = self._best_safe_point().item() + delta
            candidate = np.clip(candidate, *DOMAIN[0]).reshape(1, -1)

        return candidate

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self._acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for i in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(
                DOMAIN.shape[0]
            )
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN, approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        if not np.any(self._v_upper(np.array(x_values)) < SAFETY_THRESHOLD):
            self.is_hard = True

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return np.array([x_opt]).reshape(1, -1)

    def _v_upper(self, x: np.ndarray):
        mean, std = self.v_hat.predict(x, return_std=True)  # type: ignore
        mean = mean + V_MEAN_PRIOR
        return norm.interval(self.v_confidence_level, loc=mean, scale=std)[1]

    def _v_lower(self, x: np.ndarray):
        mean, std = self.v_hat.predict(x, return_std=True)  # type: ignore
        mean = mean + V_MEAN_PRIOR
        return norm.interval(self.v_confidence_level, loc=mean, scale=std)[0]

    def _f_upper(self, x: np.ndarray):
        mean, std = self.f_hat.predict(x, return_std=True)  # type: ignore
        return norm.interval(self.f_confidence_level, loc=mean, scale=std)[1]

    def _best_safe_point(self):
        safe_indices = np.where(self.vs < SAFETY_THRESHOLD)[0]
        return self.xs[safe_indices[np.argmax(self.fs[safe_indices])]].reshape(1, -1)

    def _unsafety_penalty(self, x: np.ndarray):
        v_upper = self._v_upper(x)
        return self.safety_constraint_coef * max(v_upper - SAFETY_THRESHOLD, 0)

    def _general_expected_improvement(self, x: np.ndarray):
        f_mean, f_std = self.f_hat.predict(x, return_std=True)  # type: ignore
        improvement = f_mean - np.max(self.fs)
        z_f = improvement / f_std

        reward = improvement * norm.cdf(z_f) + f_std * norm.pdf(z_f)
        penalty = self.safety_coef * self._v_upper(x)

        return self.reward_coef * reward - penalty - self._unsafety_penalty(x)

    def _general_ucb(self, x: np.ndarray):
        reward = self.reward_coef * self._f_upper(x)
        penalty = self.safety_coef * self._v_upper(x)

        return reward - penalty - self._unsafety_penalty(x)

    def _acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)

        if self.acquisition_function == "general_ucb":
            return self._general_ucb(x)
        elif self.acquisition_function == "general_ei":
            return self._general_expected_improvement(x)

        raise ValueError(f"Unknown acquisition function {self.acquisition_function}")

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        self.xs = np.vstack([self.xs, x])
        self.fs = np.vstack([self.fs, f])
        self.vs = np.vstack([self.vs, v])

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        return self._best_safe_point()

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass
