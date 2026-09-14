
import torch


class UncertaintyEstimator:
    """
    Semantic uncertainty estimator.

    u = alpha * (1 - max_similarity) + beta * normalized_entropy

    EMA:
    u_ema = mu * u_previous + (1 - mu) * u_current
    """

    def __init__(self, alpha=1.0, beta=1.0, ema_mu=0.8):
        self.alpha = alpha
        self.beta = beta
        self.ema_mu = ema_mu
        self.previous = {}

    @staticmethod
    def normalized_entropy(similarities):
        probs = torch.softmax(similarities, dim=-1)

        entropy = -(
            probs * torch.log(probs.clamp_min(1e-12))
        ).sum(dim=-1)

        num_classes = similarities.shape[-1]

        if num_classes <= 1:
            return torch.zeros_like(entropy)

        max_entropy = torch.log(
            torch.tensor(
                float(num_classes),
                device=similarities.device,
                dtype=similarities.dtype
            )
        )

        return entropy / max_entropy.clamp_min(1e-12)

    def compute(self, similarities):
        if similarities is None or similarities.numel() == 0:
            return torch.empty(
                0,
                device=(
                    similarities.device
                    if similarities is not None
                    else "cpu"
                )
            )

        max_similarity = similarities.max(dim=-1).values

        entropy = self.normalized_entropy(similarities)

        uncertainty = (
            self.alpha * (1.0 - max_similarity)
            + self.beta * entropy
        )

        return uncertainty.clamp(0.0, 1.0)

    def update(self, track_ids, uncertainty):

        result = []

        for track_id, u in zip(track_ids, uncertainty):

            track_id = int(track_id)
            current = float(u.detach().cpu())

            if track_id in self.previous:
                current = (
                    self.ema_mu * self.previous[track_id]
                    + (1.0 - self.ema_mu) * current
                )

            self.previous[track_id] = current
            result.append(current)

        if not result:
            return uncertainty

        return torch.tensor(
            result,
            dtype=uncertainty.dtype,
            device=uncertainty.device
        )

    def reset(self):
        self.previous.clear()
