import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import os
import pandas as pd
import sys
import statsmodels.api as sm
from scipy.stats import entropy
import scipy.stats as stats
import glob
import tempfile

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

def reward(df):
    '''Models the monetary reward as points'''
    # The BART task has no monetary penalty, so any negative value should be converted to 0
    return df.diff().fillna(0)

def mean_RT(rt_data):
    '''Filters out outliers and calculates the mean reaction times'''
    Q1 = np.percentile(rt_data, 25)
    Q3 = np.percentile(rt_data, 75)
    IQR = Q3 - Q1
    mask = (rt_data >= Q1 - 1.5*IQR) & (rt_data <= Q3 + 1.5*IQR)
    return rt_data[mask].mean()

def calculate_KLD(free, ctrl, bins=10, eps=1e-10):
    '''Returns Kullback-Leibler divergence'''
    histfree, edges = np.histogram(free, bins=bins, density=False)  # Note: density=False to match MATLAB
    histctrl, edges = np.histogram(ctrl, bins=bins, density=False)
    
    # Normalize histograms manually to match MATLAB behavior
    freeHist_norm = histfree / np.sum(histfree)
    ctrlHist_norm = histctrl / np.sum(histctrl)
    
    # Calculate KLD exactly as in MATLAB
    d = np.sum(
        freeHist_norm * np.log2(freeHist_norm + eps) - 
        freeHist_norm * np.log2(ctrlHist_norm + eps)
    )
    
    # This line of code fixes values to match Rhiannon's paper
    d = -1 * np.log10(d)

    return d


# Regression Fit RT
def RT_regression(ax):
    '''Uses the statsmodels module to fit a regression line to a scatter plot'''
    # Get scatter plot data from the axis using correct matplotlib class
    scatter_plots = [child for child in ax.get_children() if isinstance(child, PathCollection)]
    
    # Initialize lists to store all data points
    x_all = []
    y_all = []
    
    # Collect all data points from scatter plots
    for scatter in scatter_plots:
        points = scatter.get_offsets()
        x_all.extend(points[:, 0])
        y_all.extend(points[:, 1])
    
    # Convert to numpy arrays
    x_data = np.array(x_all)
    y_data = np.array(y_all)
    
    # Fit regression
    X = sm.add_constant(x_data)
    model = sm.OLS(y_data, X).fit()
    
    # Create prediction line
    x_range = np.linspace(min(x_data), max(x_data), 100)
    X_pred = sm.add_constant(x_range)
    y_pred = model.predict(X_pred)

    # Get prediction confidence interval
    pred = model.get_prediction(X_pred)
    pred_summary = pred.summary_frame(alpha=0.05)  # 95% confidence interval
    
    # Plot regression line and confidence intervals
    ax.plot(x_range, y_pred, 'r-', alpha=0.5)
    ax.fill_between(x_range, pred_summary['obs_ci_lower'], pred_summary['obs_ci_upper'], color='gray', alpha=0.2)

    # F-statistic
    ax.text(x_range[int(len(x_range)/2)], 800, f"F-stat={round(model.fvalue,2)}", ha='center', va='bottom')

def extract(task):
    '''An efficient method to extract the balloon subsets from the larger batch dataframe'''
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

# Main function passes into dashboard.py; To run local files, run directly.

def main(taskdf):
    '''Takes in a dictionary of multiple trial dataframes; returns batch_analysis.png including all batch analysis figures'''

    print("Running analysis for entire trial batch...")

    # Use apply() to process each DataFrame in the dictionary
    struct = {key: extract(task) for key, task in taskdf.items()}

    # Calculate overall statistics for the text information section
    total_subjects = len(struct)
    mean_accuracy = np.mean([data['acc'] for task, data in struct.items()]) * 100
    mean_points = np.mean([data['free']['total reward'].iloc[-1] for task, data in struct.items()])
    mean_rt = np.mean([mean_RT(data['free']['reactionTime(ms)']) for task, data in struct.items()])
    mean_kld = np.mean([calculate_KLD(data['ctrls']['inflationTime(ms)'], data['free']['inflationTime(ms)']) 
                        for task, data in struct.items()])
    
    # Setup figure with no auto layout manager
    fig = plt.figure(figsize=(12, 15), layout=None)
    
    # Add text information section at the top
    text_ax = fig.add_axes([0.1, 0.92, 0.8, 0.06])  # [left, bottom, width, height]
    text_ax.axis('off')  # Turn off the axis
    
    # Add text information
    text_ax.text(0.0, 0.9, f"BART Batch Analysis - {total_subjects} Subjects", fontsize=12, fontweight='bold')
    text_ax.text(0.0, 0.6, f"Mean Accuracy: {mean_accuracy:.2f}%", fontsize=10)
    text_ax.text(0.0, 0.3, f"Mean Points: {mean_points:.2f}    Mean RT: {mean_rt:.2f}ms    Mean KLD: {mean_kld:.4f}", fontsize=10)
    
    # Add horizontal line to separate text from plots
    text_ax.axhline(y=0.0, color='black', linestyle='-', alpha=0.5)
    
    # Adjust the gridspec to start below the text section
    # Additional width ration because the default caused overlapping
    gs = fig.add_gridspec(6, 3, top=0.9, hspace=0.4, wspace=0.3)  # Adjusted top to leave space for text

    ax11 = fig.add_subplot(gs[0:2, 0])
    ax12 = fig.add_subplot(gs[2:4, 0])
    ax13 = fig.add_subplot(gs[4:6, 0])
    ax13.set_xlabel('KLD - Active/Passive', fontsize=10.5, fontweight='bold')

    ax11.set_ylabel('Total Points (Reward)', fontsize=10.5, fontweight='bold')
    ax12.set_ylabel('Accuracy', fontsize=10.5, fontweight='bold')
    ax13.set_ylabel('RT', fontsize=10.5, fontweight='bold')

    ax21 = fig.add_subplot(gs[0:2, 1])
    ax22 = fig.add_subplot(gs[2:4, 1])
    ax23 = fig.add_subplot(gs[4:6, 1])
    ax23.set_xlabel('T-stat - Active/Passive', fontsize=10.5, fontweight='bold')

    ax32 = fig.add_subplot(gs[3:5, 2])
    ax32.set_title('Reward vs Accuracy across trials', fontsize=10.5, fontweight='bold')
    ax32.set_xlabel('Accuracy', fontsize=10.5, fontweight='bold')
    ax32.set_ylabel('Total Points (Reward)', fontsize=10.5, fontweight='bold')

    axhist = fig.add_subplot(gs[1:3, 2])
    axhist.set_title('KLD Histogram', fontsize=10.5, fontweight='bold')

    kldhist = []

    # Figures
    for task, data in struct.items():
        kld = calculate_KLD(data['ctrls']['inflationTime(ms)'], data['free']['inflationTime(ms)'])
        kldhist.append(kld)
        totalpoints = data['free']['total reward'].iloc[-1]

        # Using scipy's built in t-stat func
        t_stat, p_value = stats.ttest_ind(data['free']['inflationTime(ms)'], data['ctrls']['inflationTime(ms)'])  # T-statistic for ITs?

        ax11.scatter(kld, totalpoints, color='black')
        ax12.scatter(kld, data['acc'], color='black')
        ax13.scatter(kld, mean_RT(data['free']['reactionTime(ms)']), color='black')
        
        ax21.scatter(t_stat, totalpoints, color='black')
        ax22.scatter(t_stat, data['acc'], color='black')
        ax23.scatter(t_stat, mean_RT(data['free']['reactionTime(ms)']), color='black')

        ax32.scatter(data['acc'], data['free']['total reward'].iloc[-1])

    axhist.hist(kldhist, bins=10, density=True)

    # Set axis limits
    def get_plot_limits(ax):
        scatter_plots = [child for child in ax.get_children() if isinstance(child, PathCollection)]
        x_values = []
        
        for scatter in scatter_plots:
            points = scatter.get_offsets()
            if len(points) > 0:
                x_values.extend(points[:, 0])
                
        return min(x_values) if x_values else None, max(x_values) if x_values else None

    # Get min/max for KLD column (ax11, ax12, ax13)
    kld_mins, kld_maxs = [], []
    for ax in [ax11, ax12, ax13]:
        min_val, max_val = get_plot_limits(ax)
        if min_val is not None:
            kld_mins.append(min_val)
            kld_maxs.append(max_val)

    # Get min/max for t-stat column (ax21, ax22, ax23)
    tstat_mins, tstat_maxs = [], []
    for ax in [ax21, ax22, ax23]:
        min_val, max_val = get_plot_limits(ax)
        if min_val is not None:
            tstat_mins.append(min_val)
            tstat_maxs.append(max_val)

    # Set limits for KLD column
    if kld_mins and kld_maxs:
        for ax in [ax11, ax12, ax13]:
            ax.set_xlim(min(kld_mins), max(kld_maxs))

    # Set limits for t-stat column
    if tstat_mins and tstat_maxs:
        for ax in [ax21, ax22, ax23]:
            ax.set_xlim(min(tstat_mins), max(tstat_maxs))

            

    # RT Regression - for divergence and t stat
    modeld = RT_regression(ax13)
    modelt = RT_regression(ax23)
    #ax13.plot([], [])   
    #ax23.plot([], [])


    for ax in fig.get_axes():
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)

    # TODO:
    # Mean accuracy (overall, across each color)
    # Mean points
    # Mean RT

    output_path = get_output_path("batch_analysis.png")
    fig.savefig(output_path)
    return fig, output_path


if __name__=="__main__":
    # Running directly returns local batch analysis figures
    PATH='/home/ethant/Documents/Projects/bart/sessions/'
    taskfiles = glob.glob(PATH + '**/task_data*.xlsx', recursive=True)

    taskdf = {}

    for file in taskfiles:
        filename = os.path.splitext(os.path.basename(file))[0]

        task = pd.read_excel(file)  # specify sheet name if needed: sheet_name='Sheet1'
        taskdf[filename] = task

    main(taskdf)
