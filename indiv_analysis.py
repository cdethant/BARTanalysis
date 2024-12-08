import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
import statistics as s


### Functions-----------------------------------------
# Remove Outliers
def outlier(df):
    IQR = df.quantile(0.75) - df.quantile(0.25)
    df = df[(df < (df.quantile(0.75) + 1.5*IQR)) & (df > (df.quantile(0.25) - 1.5*IQR))]
    return df

# Models the monetary reward as points
# The BART task has no monetary penalty,
# so any negative value should be converted to 0
def reward(df):
    return df.diff().fillna(0)

def KLD(free, ctrl, bins=10, eps=1e-10):
    histfree, edges = np.histogram(free,bins=10,density=True)
    histctrl, edges = np.histogram(ctrl, bins=10, density=True)

    # Add a small epsilon to avoid log(0)
    histfree = np.clip(histfree, eps, 1)
    histctrl = np.clip(histctrl, eps, 1)

    # scipy's entropy uses the same formula as KLD
    kld = entropy(histfree, histctrl)
    return kld
### ---------------------------------------------------

# Specify folder path
PATH='/home/ethant/Projects/bart/'

taskfile = filedialog.askopenfilename(initialdir=PATH, title="Select a file")
sessionID = re.search(r'\d+_\d+_\d+_\d+_\d+_\d+', taskfile).group()

taskdf = pd.read_excel(taskfile, header=0)
taskdf.to_csv()

# Control Trials
ctrls = taskdf[taskdf['balloonType'].str.contains('special')]
free = taskdf[~taskdf['balloonType'].str.contains('special') & ~taskdf['balloonType'].str.contains('gray')] 

totalballoons = taskdf.shape[0]
totalctrls = ctrls.shape[0]


# Colors
reds = taskdf[taskdf['balloonType'].str.contains('red')]
oranges = taskdf[taskdf['balloonType'].str.contains('orange')]
yellows = taskdf[taskdf['balloonType'].str.contains('yellow')]

points = taskdf['total reward']
pointsRed = reward(points[taskdf['balloonType'].str.contains('red')])
pointsOrange = reward(points[taskdf['balloonType'].str.contains('orange')])
pointsYellow = reward(points[taskdf['balloonType'].str.contains('yellow')])


# Reaction Times - outliers removed
ctrlRTs = outlier(ctrls['reactionTime(ms)'])
freeRTs = outlier(free['reactionTime(ms)'])

redRTS = outlier(reds['reactionTime(ms)'])
orangeRTs = outlier(oranges['reactionTime(ms)'])
yellowRTs = outlier(yellows['reactionTime(ms)'])

# post pop RTs are the RTs that follow reward = 0
postbankRTs = outlier(freeRTs[reward(points) != 0])
postpopRTs = outlier(freeRTs[reward(points) == 0])

# Inflation Times
ITs = taskdf['inflationTime(ms)']
ctrlITs = ctrls['inflationTime(ms)']
freeITs = free['inflationTime(ms)']


banked = taskdf.loc[taskdf['outcome'] == 'banked']
popped = taskdf.loc[taskdf['outcome'] == 'popped']


# Accuracies - Overall & Color
accTotal = (banked.shape[0] / totalballoons) * 100
accRed = (banked[banked['balloonType'].str.contains('red')].shape[0] / taskdf[taskdf['balloonType'].str.contains('red')].shape[0]) * 100
accOrange = (banked[banked['balloonType'].str.contains('orange')].shape[0] / taskdf[taskdf['balloonType'].str.contains('orange')].shape[0]) * 100
accYellow = (banked[banked['balloonType'].str.contains('yellow')].shape[0] / taskdf[taskdf['balloonType'].str.contains('yellow')].shape[0]) * 100

avgReward = np.mean(reward(points))
totalReward = np.sum(reward(points))

#if p<0.01 ==> contingency table 
# pandas crosstab function requires an aggfunc: count, mean, var etc.
# which is matlabs default crosstab function?
#crosstab = pd.crosstab(banked[banked['balloonType'].str.contains('red')],banked[banked['balloonType'].str.contains('orange')],banked[banked['balloonType'].str.contains('yellow')])
#print(crosstab)

# if p >, no significant difference.

# distance of active trial distribution from passive trial distribution.
# So lower divergence values means more similar distributions
# => less impulsive choosers.


# The number of Histogran bins.
# N=10 for K-L Divergence Metric
n_bins = 10


KLDred = KLD(freeITs[free['balloonType'].str.contains('red')], ctrlITs[ctrls['balloonType'].str.contains('red')])

#KLDred = KLDImpulse(freeITs[free['balloonType'].str.contains('red')],ctrlITs[ctrls['balloonType'].str.contains('red')])
print(KLDred)

# Kruskal Wallis Test


# FIGURES ---------------------------------------------------

# Create a figure and axes
fig = plt.figure(layout='constrained')
gs = fig.add_gridspec(2,6)


ax1 = fig.add_subplot(gs[0,3:])
ax2 = fig.add_subplot(gs[1,0:3])

## FIGURE 1
ax0 = fig.add_subplot(gs[0,0:3])
ax0.hist(freeITs, bins=n_bins, alpha=0.5, edgecolor='black', color='blue')
ax0.hist(ctrlITs, bins=n_bins, alpha=0.5, edgecolor='black', color='green')
ax0.set_title('ITs - active vs passive')
ax0.set_xlabel('time (s)')
ax0.set_ylabel('IT count')

## FIGURE 2 - ITs for each balloon color
ax1.hist(ITs[taskdf['balloonType'].str.contains('red')], bins=n_bins, alpha=0.5, edgecolor='black', color='red')
ax1.hist(ITs[taskdf['balloonType'].str.contains('orange')], bins=n_bins, alpha=0.5, edgecolor='black', color='orange')
ax1.hist(ITs[taskdf['balloonType'].str.contains('yellow')], bins=n_bins, alpha=0.5, edgecolor='black', color='yellow')
ax1.set_title('ITs - colors')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('IT count')

## FIGURE 3 - rewards from active trials
pointsRedActive = reward(points[taskdf['balloonType'].str.contains('red') & ~taskdf['balloonType'].str.contains('special')])
pointsOrangeActive = reward(points[taskdf['balloonType'].str.contains('orange') & ~taskdf['balloonType'].str.contains('special')])
pointsYellowActive = reward(points[taskdf['balloonType'].str.contains('yellow') & ~taskdf['balloonType'].str.contains('special')])

ax2.hist(pointsRedActive, bins=n_bins, alpha=0.5, edgecolor='black', color='red')
ax2.hist(pointsOrangeActive, bins=n_bins, alpha=0.5, edgecolor='black', color='orange')
ax2.hist(pointsYellowActive, bins=n_bins, alpha=0.5, edgecolor='black', color='yellow')
ax2.set_title('Rewards across color')
ax2.set_xlabel('reward (points)')
ax2.set_ylabel('count')

## FIGURE 4 - Violin plots
ax3 = fig.add_subplot(gs[1,3])
ax4 = fig.add_subplot(gs[1,4])
ax5 = fig.add_subplot(gs[1,5])

ax3.violinplot([ctrlRTs, freeRTs], showmedians=True, positions=[1,2])
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Control', 'Free'], rotation=45)

ax4.violinplot([redRTS, yellowRTs, orangeRTs], showmedians=True, positions=[1,2,3])
ax4.set_xticks([1, 2, 3])
ax4.set_xticklabels(['Red', 'Yellow', 'Orange'], rotation=45)

ax5.violinplot([postbankRTs, postpopRTs], showmedians=True, positions=[1,2])
ax5.set_xticks([1, 2])
ax5.set_xticklabels(['Post Bank', 'Post Pop'], rotation=45)

for ax in fig.get_axes():
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)

plt.show()

# ----------------------------------------------------------

# Save PDF
fig.savefig('session_' + sessionID) # TODO: Replace with "{patient/sessionID}"
c = canvas.Canvas('session_' + sessionID + '.pdf', pagesize=letter)

c.setFont("Helvetica", 12)
c.drawString(60, 750, f"BART analysis - Session {sessionID}")

c.drawString(60, 720, f"Total Balloons: {totalballoons}")
c.drawString(60, 690, f"Overall Accuracy: {accRed}")
c.drawString(60, 670, f"Accuracy Red: {accRed}")
c.drawString(60, 650, f"Accuracy Orange: {accOrange}")
c.drawString(60, 630, f"Accuracy Yellow: {accYellow}")
c.drawString(60, 610, "Signifcant difference in accuracy among balloon colors: X^2")

c.drawString(60, 580, f"Total Reward: {totalReward}    Average Reward: {avgReward}")
c.drawString(60, 560, f"KLD-derived Impulsivity: {KLDred}")

c.drawImage(f'session_{sessionID}.png', 50, 100, width=500, height=375)

c.save()