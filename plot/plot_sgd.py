import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipdb


filename = './data/fashion_mnist_lenet_sgd.csv'

df = pd.read_csv(filename)

fig = plt.figure()
fig.set_size_inches(w=10, h=6, forward=True)
fig.set_dpi(100)
ax = plt.gca()

for p in np.sort(df.posterior_mean_stopping_error_train.unique()):
    dfp = df.loc[df.posterior_mean_stopping_error_train == p]
    ax.plot(dfp.alpha, dfp.error_bound, 'x-', label=f'posterior_mean_stopping_error_train={p:.2f}')

ax.set_xlabel('alpha', fontsize=12)
ax.set_ylabel('error_bound', fontsize=12)
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, 1.3])
ax.set_title('fashion_mnist_lenet_sgd')
ax.grid()
ax.legend()
# plt.show()

plt.savefig('./output/fashion_mnist_lenet_sgd.pdf')



filename = './data/mnist_lenet_sgd.csv'

df = pd.read_csv(filename)

fig = plt.figure()
fig.set_size_inches(w=10, h=6, forward=True)
fig.set_dpi(100)
ax = plt.gca()

for p in np.sort(df.posterior_mean_stopping_error_train.unique()):
    dfp = df.loc[df.posterior_mean_stopping_error_train == p]
    ax.plot(dfp.alpha, dfp.error_bound, 'x-', label=f'posterior_mean_stopping_error_train={p:.3f}')

ax.set_xlabel('alpha', fontsize=12)
ax.set_ylabel('error_bound', fontsize=12)
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, 0.8])
ax.set_title('mnist_lenet_sgd')
ax.grid()
ax.legend()
# plt.show()

plt.savefig('./output/mnist_lenet_sgd.pdf')

