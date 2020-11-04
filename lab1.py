import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import math

def normal(n):
    result = np.random.standard_normal(n)
    return result


def uniform(n, a, b):
    return np.random.uniform(a, b, size=n)


def hiSquare(dist, k, A, B, N):
    delta = np.linspace(A, B, num=k)
    array = np.array([stats.norm.cdf(delta[0])])
    q = np.array([len(dist[dist <= delta[0]])])
    for i in range(0, len(delta) - 1):
        new_ar = stats.norm.cdf(delta[i + 1]) - stats.norm.cdf(delta[i])
        array = np.append(array, new_ar)
        q = np.append(q, len(dist[(dist <= delta[i + 1]) & (dist >= delta[i])]))
    array = np.append(array, 1 - stats.norm.cdf(delta[-1]))
    q = np.append(q, len(dist[dist >= delta[-1]]))
    result = np.divide(np.multiply((q - N * array), (q - N * array)), array * N)

    print("n_i")
    print(q)

    print("p_i")
    print(array)

    print("n*p_i")
    print(array*N)

    print("n_i-n*p_i")
    print(q-array*N)

    print("result")
    print(result)

    return np.around(np.sum(result), decimals=2)


def drawHistogram(data, name):
    count_interval = 1 + math.floor(math.log2(len(data)))

    f, ax = plt.subplots(1, 1)
    ax.hist(data, bins=count_interval)
    ax.set_title(name)

    f.tight_layout()
    plt.show()

alpha = 0.05
k = 6
A = -2
B = 2
N = 160

distNorm = normal(N)
distUniform = uniform(N, A, B)

value = stats.chi2.ppf(1 - alpha, k - 3)
print("\n\nNORM")
hiNorm = hiSquare(distNorm, k - 3, A, B, N)
print("\n\nUNIFORM")
hiUniform = hiSquare(distUniform, k - 3, A, B, N)

print("\n\nSummary:")
print("quantile = " + str(value))
print("hiNorm = " + str(hiNorm))
print("hiUniform = " + str(hiUniform))


drawHistogram(distNorm, "NORM")
drawHistogram(distUniform, "UNIFORM")
