import base64
import io
import sys

import matplotlib.pyplot as plt
import numpy as np


def cos_sim(x, y):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    return (x * y).sum() / (x_norm * y_norm)


def sample(dim: int):
    u = np.random.randn(dim)
    v = np.random.randn(dim)
    w = np.random.randn(dim)

    d = v - u
    u_norm_sq = np.linalg.norm(u) ** 2
    d_norm_sq = np.linalg.norm(d) ** 2

    alpha = np.dot(d, w) * u_norm_sq - np.dot(d, u) * np.dot(u, w)
    alpha = alpha / (np.dot(u, w) * d_norm_sq - np.dot(d, u) * np.dot(d, w))

    x_alpha = alpha * v + (1 - alpha) * u

    return alpha, cos_sim(x_alpha, w)


def ghostty_show():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")

    # Send the entire image as a single payload (m=0 is implied)
    sys.stdout.write(f"\033_Ga=T,f=100;{data}\033\\\n")
    plt.close()


def plot(n: int, dim: int):
    samples = [sample(dim) for i in range(n)]

    x_samples = [s[0] for s in samples]
    y_samples = [s[1] for s in samples]

    x_samples = np.array(x_samples)
    x_samples = x_samples[
        (x_samples <= np.quantile(x_samples, 0.95))
        & (x_samples >= np.quantile(x_samples, 0.05))
    ]
    plt.hist(x_samples)
    ghostty_show()

    plt.hist(y_samples)
    ghostty_show()


if __name__ == "__main__":
    n = 10000
    dim = 10000
    plot(n, dim)
