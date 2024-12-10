import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import entropy
import scipy.stats as stats
import glob

# Get Points
def reward(df):
    return df.diff().fillna(0)

# KLD compute
def KLD(free, ctrl, bins=10, eps=1e-10):
    histfree, edges = np.histogram(free,bins=10,density=True)
    histctrl, edges = np.histogram(ctrl, bins=10, density=True)

    # Add a small epsilon to avoid log(0)
    histfree = np.clip(histfree, eps, 1)
    histctrl = np.clip(histctrl, eps, 1)

    # scipy's entropy uses the same formula as KLD
    kld = entropy(histfree, histctrl)
    return kld

# More efficient method of extracting subsets from each dataframe
def extract(task):
    mask_special = task['balloonType'].str.contains('special')
    mask_gray = task['balloonType'].str.contains('gray')
    mask_red = task['balloonType'].str.contains('red')
    mask_orange = task['balloonType'].str.contains('orange')
    mask_yellow = task['balloonType'].str.contains('yellow')
    
    # Control Trials
    ctrls = task[mask_special]
    free = task[~mask_special & ~mask_gray]
    
    totalballoons = task.shape[0]
    totalctrls = ctrls.shape[0]
    accuracy = task.loc[task['outcome'] == 'banked'].shape[0]/totalballoons
    
    # Colors
    reds = task[mask_red]
    oranges = task[mask_orange]
    yellows = task[mask_yellow]
    
    # Return a dictionary of results for this DataFrame
    return {
        'totalballoons': totalballoons,
        'ctrls': ctrls,
        'free': free,
        'totalctrls': totalctrls,
        'reds': reds,
        'oranges': oranges,
        'yellows': yellows,
        'acc': accuracy
    }

# Batch Analysis
PATH='/home/ethant/Projects/bart/'
taskfiles = glob.glob(PATH + '**/task_data*.xlsx', recursive=True)

taskdf = {}

for file in taskfiles:
    filename = os.path.splitext(os.path.basename(file))[0]

    task = pd.read_excel(file)  # specify sheet name if needed: sheet_name='Sheet1'
    taskdf[filename] = task


# taskdf.to_csv()

# Use apply() to process each DataFrame in the dictionary
struct = {key: extract(task) for key, task in taskdf.items()}


# Setup of plots
fig = plt.figure(layout='constrained')
gs = fig.add_gridspec(3,3)

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[0,1])

# First Column
for task, data in struct.items():
    kld = KLD(data['ctrls']['inflationTime(ms)'], data['free']['inflationTime(ms)'])
    totalpoints = data['free']['total reward'].iloc[-1]
    ax0.scatter(kld, totalpoints, marker='x', color='blue', s=50, linewidth=2)
    ax1.scatter(kld, data['acc'], marker='x', color='blue', s=50, linewidth=2)
    
# Using scipy's built in t-stat func
for task, data in struct.items():
    t_stat, p_value = stats.ttest_ind(data['free']['inflationTime(ms)'], data['ctrls']['inflationTime(ms)'])  # T-statistic for ITs?
    totalpoints = data['free']['total reward'].iloc[-1]
    ax2.scatter(totalpoints,t_stat)

# Scatterplot + Histogram


for ax in fig.get_axes():
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)


plt.show()

# Dont need KLD Impulse?