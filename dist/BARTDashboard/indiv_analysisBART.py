import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import tkinter as tk
from tkinter import filedialog
import os
import sys
import tempfile
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import pandas as pd
from scipy.stats import entropy
import statistics as s

### Functions-----------------------------------------
def get_output_path(filename):
    '''Returns a temp directory (in deployment) or a subfolder (in development) to create and store BART analysis figures'''
    # For development
    if not getattr(sys, 'frozen', False):
        # Create figures subfolder in current directory if it doesn't exist
        os.makedirs("figures", exist_ok=True)
        return os.path.join("figures", filename)
    
    # For bundled app (.exe)
    else:
        # Use a figures subfolder in temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), "figures")
        os.makedirs(temp_dir, exist_ok=True)
        return os.path.join(temp_dir, filename)
    

# Remove Outliers
def outlier(df):
    '''Find outliers for exclusion'''
    IQR = df.quantile(0.75) - df.quantile(0.25)
    df = df[(df < (df.quantile(0.75) + 1.5*IQR)) & (df > (df.quantile(0.25) - 1.5*IQR))]
    return df

def reward(df):
    '''Models the monetary reward as points'''
    # The BART task has no monetary penalty, so any negative value should be converted to 0
    return df.diff().fillna(0)

def KLD(free, ctrl, bins=10, eps=1e-10):
    '''Returns Kullback-Leibler divergence'''
    histfree, edges = np.histogram(free,bins=10,density=True)
    histctrl, edges = np.histogram(ctrl, bins=10, density=True)

    # Add a small epsilon to avoid log(0)
    histfree = np.clip(histfree, eps, 1)
    histctrl = np.clip(histctrl, eps, 1)

    # scipy's entropy uses the same formula as KLD
    kld = entropy(histfree, histctrl)
    return kld

# ----------------------------------------------------------

def main(trialdf, sessionID):
    '''Takes in a pandas dataframe of individual trial data and its ID; returns {sessionID}.png including all analysis figures'''

    print(f"Running analysis for trial session {sessionID}...")

    # Control Trials
    ctrls = trialdf[trialdf['balloonType'].str.contains('special')]
    free = trialdf[~trialdf['balloonType'].str.contains('special') & ~trialdf['balloonType'].str.contains('gray')] 

    totalballoons = trialdf.shape[0]
    totalctrls = ctrls.shape[0]


    # Colors
    reds = trialdf[trialdf['balloonType'].str.contains('red')]
    oranges = trialdf[trialdf['balloonType'].str.contains('orange')]
    yellows = trialdf[trialdf['balloonType'].str.contains('yellow')]

    points = trialdf['total reward']
    all_rewards = reward(points)

    pointsRed = all_rewards[trialdf['balloonType'].str.contains('red')]
    pointsOrange = all_rewards[trialdf['balloonType'].str.contains('orange')]
    pointsYellow = all_rewards[trialdf['balloonType'].str.contains('yellow')]

    # Remove zeros
    pointsRed = pointsRed[pointsRed != 0]
    pointsOrange = pointsOrange[pointsOrange != 0]
    pointsYellow = pointsYellow[pointsYellow != 0]


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
    ITs = trialdf['inflationTime(ms)']
    ctrlITs = ctrls['inflationTime(ms)']
    freeITs = free['inflationTime(ms)']


    banked = trialdf.loc[trialdf['outcome'] == 'banked']
    popped = trialdf.loc[trialdf['outcome'] == 'popped']


    # Accuracies - Overall & Color
    accTotal = (banked.shape[0] / totalballoons) * 100
    accRed = (banked[banked['balloonType'].str.contains('red')].shape[0] / trialdf[trialdf['balloonType'].str.contains('red')].shape[0]) * 100
    accOrange = (banked[banked['balloonType'].str.contains('orange')].shape[0] / trialdf[trialdf['balloonType'].str.contains('orange')].shape[0]) * 100
    accYellow = (banked[banked['balloonType'].str.contains('yellow')].shape[0] / trialdf[trialdf['balloonType'].str.contains('yellow')].shape[0]) * 100

    avgReward = np.mean(reward(points))
    totalReward = np.sum(reward(points))

    #if p<0.01 ==> contingency table 
    # pandas crosstab function requires an aggfunc: count, mean, var etc.
    # which is matlabs default crosstab function?
    #crosstab = pd.crosstab(banked[banked['balloonType'].str.contains('red')],banked[banked['balloonType'].str.contains('orange')],banked[banked['balloonType'].str.contains('yellow')])
    #print(crosstab)

    # if p >, no significant difference.

    # The number of Histogran bins.
    # N=10 for K-L Divergence Metric
    n_bins = 10


    KLDred = KLD(freeITs[free['balloonType'].str.contains('red')], ctrlITs[ctrls['balloonType'].str.contains('red')])

    # Create a figure and axes
    fig = plt.figure(figsize=(12,15),layout=None)

    # First create a section for the text at the top
    # Text axes
    text_ax = fig.add_axes([0.1, 0.85, 0.8, 0.1])  # [left, bottom, width, height]
    text_ax.axis('off')  # Turn off the axis
    
    # Add text information
    text_ax.text(0.0, 0.9, f"BART analysis - Session {sessionID}", fontsize=12, fontweight='bold')
    text_ax.text(0.0, 0.8, f"Total Balloons: {totalballoons}", fontsize=10)
    text_ax.text(0.0, 0.7, f"Overall Accuracy: {accTotal:.2f}%", fontsize=10)
    text_ax.text(0.0, 0.6, f"Accuracy Red: {accRed:.2f}%", fontsize=10)
    text_ax.text(0.0, 0.5, f"Accuracy Orange: {accOrange:.2f}%", fontsize=10)
    text_ax.text(0.0, 0.4, f"Accuracy Yellow: {accYellow:.2f}%", fontsize=10)
    text_ax.text(0.0, 0.3, f"Significant difference in accuracy among balloon colors: X^2", fontsize=10)
    text_ax.text(0.0, 0.2, f"Total Reward: {totalReward:.2f}    Average Reward: {avgReward:.2f}", fontsize=10)
    text_ax.text(0.0, 0.1, f"KLD-derived Impulsivity: {KLDred:.4f}", fontsize=10)
    
    # Add horizontal line to separate text from plots
    text_ax.axhline(y=0.0, color='black', linestyle='-', alpha=0.5)
    
    # Create the histogram plots (adjust positions to account for text section)
    ax0 = fig.add_axes([0.1, 0.45, 0.35, 0.35])  # ITs - active vs passive
    ax1 = fig.add_axes([0.55, 0.45, 0.35, 0.35])  # ITs - colors
    ax2 = fig.add_axes([0.1, 0.05, 0.35, 0.35])   # Rewards across color
    
    # Violin plots (smaller, side-by-side)
    ax3 = fig.add_axes([0.55, 0.05, 0.1, 0.35])   # RT violins - control vs free
    ax4 = fig.add_axes([0.67, 0.05, 0.1, 0.35])   # RT violins - colors
    ax5 = fig.add_axes([0.79, 0.05, 0.1, 0.35])   # RT violins - bank vs pop

    ## FIGURE 1
    ax0.hist(freeITs, bins=n_bins, alpha=0.5, edgecolor='black', color='#4B006E')
    ax0.hist(ctrlITs, bins=n_bins, alpha=0.5, edgecolor='black', color='#808080')
    ax0.set_title('ITs - active vs passive')
    ax0.set_xlabel('time (ms)')
    ax0.set_ylabel('IT count')

    ## FIGURE 2 - ITs for each balloon color
    ax1.hist(ITs[trialdf['balloonType'].str.contains('red')], bins=n_bins, alpha=0.5, edgecolor='black', color='red')
    ax1.hist(ITs[trialdf['balloonType'].str.contains('orange')], bins=n_bins, alpha=0.5, edgecolor='black', color='orange')
    ax1.hist(ITs[trialdf['balloonType'].str.contains('yellow')], bins=n_bins, alpha=0.5, edgecolor='black', color='yellow')
    ax1.set_title('ITs - colors')
    ax1.set_xlabel('time (ms)')
    ax1.set_ylabel('IT count')

    ## FIGURE 3 - rewards from trials (including passive)
    ax2.hist(pointsRed, bins=n_bins, alpha=0.5, edgecolor='black', color='red')
    ax2.hist(pointsOrange, bins=n_bins, alpha=0.5, edgecolor='black', color='orange')
    ax2.hist(pointsYellow, bins=n_bins, alpha=0.5, edgecolor='black', color='yellow')
    ax2.set_title('Rewards across color')
    ax2.set_xlabel('reward (points)')
    ax2.set_ylabel('points bin count')

    ## FIGURE 4 - Violin plots
    ctrlRT_violins = ax3.violinplot([ctrlRTs, freeRTs], showmedians=True, showextrema=False, positions=[1, 2])
    ax3.set_ylabel("Reaction Time (ms)")
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Control', 'Free'], rotation=45)

    colorRT_violins = ax4.violinplot([redRTS, yellowRTs, orangeRTs], showmedians=True, showextrema=False, positions=[1, 2, 3])
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['Red', 'Yellow', 'Orange'], rotation=45)
    ax4.set_yticklabels([]) # Remove y ticks

    bankpop_violins = ax5.violinplot([postbankRTs, postpopRTs], showmedians=True, showextrema=False, positions=[1, 2])
    ax5.set_xticks([1, 2])
    ax5.set_xticklabels(['Post Bank', 'Post Pop'], rotation=45)
    ax5.set_yticklabels([]) # Remove y ticks

    # Merge Y-axes for the violin plots
    all_RT_data = np.concatenate([ctrlRTs, freeRTs, redRTS, yellowRTs, orangeRTs, postbankRTs, postpopRTs])
    min_RT = np.min(all_RT_data)
    max_RT = np.max(all_RT_data)
    y_padding = (max_RT - min_RT) * 0.1

    ax3.set_ylim(min_RT - y_padding, max_RT + y_padding)
    ax4.set_ylim(min_RT - y_padding, max_RT + y_padding)
    ax5.set_ylim(min_RT - y_padding, max_RT + y_padding)

    # Only way to color violin plots :(
    ctrlRT_colors = ['#808080', '#4B006E'] # Gray and Purple
    colorRT_colors = ['#FF0000', 'yellow', '#FF6E00'] # Red, Yellow, Orange
    bankpop_colors = ['#34A853', '#FF0000'] # Green and Red

    for i, pc in enumerate(ctrlRT_violins['bodies']):
        pc.set_facecolor(ctrlRT_colors[i])

    ctrlRT_violins['cmedians'].set_colors(ctrlRT_colors)

    for i, pc in enumerate(colorRT_violins['bodies']):
        pc.set_facecolor(colorRT_colors[i])

    colorRT_violins['cmedians'].set_colors(colorRT_colors)

    for i, pc in enumerate(bankpop_violins['bodies']):
        pc.set_facecolor(bankpop_colors[i])

    bankpop_violins['cmedians'].set_colors(bankpop_colors)

    for ax in fig.get_axes():
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)

    output_path = get_output_path(f"{sessionID}.png")
    fig.savefig(output_path)
    return fig, output_path


if __name__=="__main__":
    # Current folder path for convenient testing. Specify different folder path if running directly on your own machine.
    PATH='/home/ethant/Documents/Projects/bart/'

    taskfile = filedialog.askopenfilename(initialdir=PATH, title="Select a file")
    sessionID = re.search(r'\d+_\d+_\d+_\d+_\d+_\d+', taskfile).group()

    trialdf = pd.read_excel(taskfile, header=0)
    trialdf.to_csv()

    main(trialdf, sessionID)