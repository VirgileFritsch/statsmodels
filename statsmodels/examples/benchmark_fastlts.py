"""
Try LTS on a large contaminated dataset

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, 2012

import time
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.robust.least_trimmed_squares import LTS

# Exammple settings
n_samples = 1000
n_features = 10
noise_scale = 1.
outliers_scale = 5.
range_cont = np.arange(0.05, 0.50, 0.05)
n_trials = 10

groups = 3

time_lts = np.zeros((len(range_cont), n_trials))
time_lts_ls = np.zeros((len(range_cont), n_trials))
discard_lts = np.zeros((len(range_cont), n_trials))
discard_lts_ls = np.zeros((len(range_cont), n_trials))
params_lts = np.zeros((len(range_cont), n_trials, n_features))
params_lts_ls = np.zeros((len(range_cont), n_trials, n_features))
for i, cont in enumerate(range_cont):
    print cont
    for trial in range(n_trials):
        # Experiment design
        # explanatory variable
        X = 3 * np.random.rand(n_samples, 1)
        # covariables
        Z = 3 * np.random.rand(n_samples, n_features - 1)
        design_matrix = np.hstack((X, Z))
        # noise
        noise = noise_scale * np.random.randn(n_samples)
        # generate outliers
        outliers = np.random.randn(n_samples)
        mask_outliers = np.ones(n_samples, dtype=bool)
        random_selection = np.random.permutation(
            n_samples)[:(1 - cont) * n_samples]
        mask_outliers[random_selection] = False
        # mix variables, noise and outliers
        beta = np.asarray([0] + [1] * (n_features - 1))
        Y = np.dot(design_matrix, beta) \
            + (~mask_outliers) * noise \
            + mask_outliers * outliers_scale * outliers

        # Run simple LTS
        t0 = time.time()
        the_fit, support = LTS(Y, design_matrix).fit(
            random_search_options=dict(max_nstarts=500, n_keep=10))
        t1 = time.time()
        time_lts[i, trial] = t1 - t0
        # How many outliers were discarded in fit?
        ground_truth = np.nonzero(mask_outliers)[0]
        lts_outliers = np.where(~support)[0]
        overlap = 0.
        for j in lts_outliers:
            if j in ground_truth:
                overlap += 1
        discard_lts[i, trial] = 100 * overlap / ground_truth.size
        params_lts[i, trial] = the_fit.params

        # Run "large-sample" LTS (~= Fast-LTS algorithm)
        t0 = time.time()
        the_fit2, support2 = LTS(Y, design_matrix).fit_large_sample(groups)
        t1 = time.time()
        time_lts_ls[i, trial] = t1 - t0
        # How many outliers were discarded in fit?
        ground_truth = np.nonzero(mask_outliers)[0]
        lts_outliers2 = np.where(~support2)[0]
        overlap = 0.
        for j in lts_outliers2:
            if j in ground_truth:
                overlap += 1
        discard_lts_ls[i, trial] = 100 * overlap / ground_truth.size
        params_lts_ls[i, trial] = the_fit2.params

plt.figure()
plt.title("Elapsed time (p=%d, n=%d)" % (n_features, n_samples))
plt.plot(range_cont, time_lts.mean(1), color="green", label="LTS")
plt.plot(range_cont, time_lts_ls.mean(1), color="blue", label="LTS_ls")
plt.xlabel("Contamination (%%)")
plt.ylabel("Elapsed time (s)")
plt.legend()

plt.figure()
plt.title("Param estimate (p=%d, n=%d)" % (n_features, n_samples))
plt.errorbar(range_cont, np.sum((params_lts - beta) ** 2, 2).mean(1),
             yerr=np.sum((params_lts - beta) ** 2, 2).var(1),
             color="green", label="LTS")
plt.errorbar(range_cont, np.sum((params_lts_ls - beta) ** 2, 2).mean(1),
         yerr=np.sum((params_lts_ls - beta) ** 2, 2).var(1),
         color="blue", label="LTS_ls")
plt.xlabel("Contamination (%%)")
plt.ylabel("Param estimate")
plt.legend()

plt.figure()
plt.title("Outliers discarded (p=%d, n=%d)" % (n_features, n_samples))
plt.plot(range_cont, discard_lts.mean(1), color="green", label="LTS")
plt.plot(range_cont, discard_lts_ls.mean(1), color="blue", label="LTS_ls")
plt.xlabel("Contamination (%%)")
plt.ylabel("Outliers discarded (%%)")
plt.legend()

plt.show()
