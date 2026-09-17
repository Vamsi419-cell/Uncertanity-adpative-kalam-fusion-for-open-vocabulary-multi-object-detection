import numpy as np
import scipy.linalg
from ..motions.kalman_filter import KalmanFilter
from ..builder import MOTION, MOTIONS


@MOTION.register_module()
class AdaptiveKalmanFilter(KalmanFilter):
    """Uncertainty-Adaptive Kalman Filter for Open-Vocabulary MOT.
    
    Dynamically scales the measurement noise covariance matrix R based on
    instantaneous or smoothed semantic/detector uncertainty:
    
        R(u) = R_0 * phi(u)
        
    where:
    - phi(u) >= 1 is a monotonically non-decreasing scaling function
    - u in [0, 1] is the uncertainty of the matched detection
    
    Supported mapping modes:
    - 'linear': phi(u) = 1 + beta * u
    - 'exponential': phi(u) = exp(beta * u)
    - 'sigmoid': phi(u) = 1 + (max_scale - 1) / (1 + exp(-k * (u - u0)))
    
    When uncertainty is high (u -> 1):
        R increases -> Kalman Gain K decreases -> Filter relies on internal motion state.
    When uncertainty is low (u -> 0):
        R approaches R_0 -> Filter snaps to detection.
    """

    def __init__(
        self,
        center_only=False,
        mapping='linear',
        beta=3.0,
        min_scale=1.0,
        max_scale=10.0,
        sigmoid_k=10.0,
        sigmoid_u0=0.5,
        **kwargs
    ):
        super().__init__(center_only=center_only)
        self.mapping = mapping
        self.beta = float(beta)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.sigmoid_k = float(sigmoid_k)
        self.sigmoid_u0 = float(sigmoid_u0)

    def compute_r_factor(self, uncertainty):
        """Compute the measurement covariance scaling factor phi(u).
        
        Args:
            uncertainty (float or None): Scalar uncertainty in [0, 1].
            
        Returns:
            float: Strictly positive multiplier for R matrix.
        """
        if uncertainty is None:
            return 1.0
        
        # Defensive clipping against NaN / Inf / out-of-bounds
        if np.isnan(uncertainty) or np.isinf(uncertainty):
            return self.max_scale
        u = float(np.clip(uncertainty, 0.0, 1.0))

        if self.mapping == 'linear':
            factor = 1.0 + self.beta * u
        elif self.mapping == 'exponential':
            factor = np.exp(self.beta * u)
        elif self.mapping == 'sigmoid':
            factor = 1.0 + (self.max_scale - 1.0) / (
                1.0 + np.exp(-self.sigmoid_k * (u - self.sigmoid_u0))
            )
        else:
            raise ValueError(f"Unknown adaptive Kalman mapping mode: {self.mapping}")

        return float(np.clip(factor, self.min_scale, self.max_scale))

    def project(self, mean, covariance, uncertainty=None):
        """Project state distribution to measurement space with adaptive noise.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8).
            uncertainty (float, optional): Scalar uncertainty for this measurement.

        Returns:
            (ndarray, ndarray): Projected mean (4,) and covariance S = H P H^T + R(u).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        base_innovation_cov = np.diag(np.square(std))

        # Scale measurement noise covariance R by phi(u)
        scale_factor = self.compute_r_factor(uncertainty)
        adaptive_r = base_innovation_cov * scale_factor

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T))
        
        return mean, covariance + adaptive_r

    def update(self, mean, covariance, measurement, uncertainty=None):
        """Run uncertainty-adaptive Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, a, h).
            uncertainty (float, optional): Uncertainty score in [0, 1] of the measurement.

        Returns:
             (ndarray, ndarray): Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, uncertainty=uncertainty)

        # Numerically stable Cholesky solve with jitter fallback if needed
        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False)
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower),
                np.dot(covariance, self._update_mat.T).T,
                check_finite=False).T
        except (scipy.linalg.LinAlgError, ValueError):
            # Regularize diagonal in case of extreme numerical edge cases
            reg_cov = projected_cov + np.eye(projected_cov.shape[0]) * 1e-4
            kalman_gain = np.dot(
                np.dot(covariance, self._update_mat.T),
                np.linalg.pinv(reg_cov)
            )

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T))

        # Ensure covariance symmetry and positive-definiteness
        new_covariance = (new_covariance + new_covariance.T) / 2.0
        
        return new_mean, new_covariance
