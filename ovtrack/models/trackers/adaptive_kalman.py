
import numpy as np


class AdaptiveKalmanFilter:

    """
    Adaptive measurement noise.

    Linear:
        R(u) = R_base * (1 + gamma * u)

    Exponential:
        R(u) = R_base * exp(gamma * u)
    """

    def __init__(
        self,
        base_filter,
        gamma=1.0,
        mapping="linear"
    ):
        self.base_filter = base_filter
        self.gamma = gamma
        self.mapping = mapping

        if hasattr(base_filter, "R"):
            self.base_R = np.array(
                base_filter.R,
                copy=True
            )
        else:
            self.base_R = None

    def get_scale(self, uncertainty):

        u = float(
            np.clip(
                uncertainty,
                0.0,
                1.0
            )
        )

        if self.mapping == "exponential":
            return float(
                np.exp(self.gamma * u)
            )

        return 1.0 + self.gamma * u

    def get_measurement_noise(self, uncertainty):

        if self.base_R is None:
            return None

        return self.base_R * self.get_scale(
            uncertainty
        )

    def set_measurement_noise(self, uncertainty):

        if self.base_R is None:
            return

        self.base_filter.R = (
            self.get_measurement_noise(
                uncertainty
            )
        )

    def restore_base_noise(self):

        if self.base_R is not None:
            self.base_filter.R = np.array(
                self.base_R,
                copy=True
            )

    def predict(self, *args, **kwargs):
        return self.base_filter.predict(
            *args,
            **kwargs
        )

    def update(
        self,
        measurement,
        uncertainty=0.0,
        *args,
        **kwargs
    ):

        self.set_measurement_noise(
            uncertainty
        )

        result = self.base_filter.update(
            measurement,
            *args,
            **kwargs
        )

        self.restore_base_noise()

        return result

    def __getattr__(self, name):
        return getattr(
            self.base_filter,
            name
        )
