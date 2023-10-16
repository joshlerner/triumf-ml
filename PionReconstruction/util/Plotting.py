import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib import cm
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.stats as stats

from PionReconstruction.util.Models import keras_weights, keras_activations, hls_weights, hls_activations, weight_types, activation_types

plt.rcParams.update({'font.size': 14})

class Plotter:
    """ 
    A helpful plotter for the common plots needed to compare and analyze ML models
    
    Input is a plotting function and the corresponding keywords
    
    Attributes
    ----------
    
    fig : pyplot figure
        The figure produced by the given function and keywords
        
    Methods
    ----------
    
    show()
        display the figure
    save(outpath)
        save the figure to the given directory
    
    """
    def __init__(self, function, **kwargs):
        """ """
        self.fig = function(**kwargs)
        
    def show(self):
        self.fig.tight_layout()
        self.fig.show()
        
    def save(self, outpath, **kwargs):
        self.fig.savefig(outpath, **kwargs)
        
def weight_profile(model=None, hls_model=None, x=None):
    """ Produce a precision profile of the weights and activations in the HLS model """
    weight_precisions = weight_types(hls_model)
    activation_precisions = activation_types(hls_model)
    
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    
    names = []
    weights = []
    
    names, weights = keras_weights(model)
    
    names.reverse()
    weights.reverse()
    
    colors = cm.Blues(np.linspace(0, 1, len(names)))
    
    bplot = ax[0][0].boxplot(weights, vert=False, medianprops=dict(linestyle='-', color='k'),
                             showfliers=False, patch_artist=True, labels=names)
    
    fig.canvas.draw()
    
    ticks = {tick.get_text():tick.get_position()[1] for tick in ax[0][0].get_yticklabels()}
    
    for i, label in enumerate(weight_precisions['layer']):
        if label in ticks.keys():
            low = 2**weight_precisions['low'][i]
            high = 2**weight_precisions['high'][i]
            y = ticks[label]
            rectangle = Rectangle((low, y - 0.4), high - low, 0.8, fill=True, color='grey', alpha=0.2)
            ax[0][0].add_patch(rectangle)
    
    ax[0][0].set_xscale('log', base=2)
    ax[0][0].set_title('Distribution of weights before optimization')
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    names, weights = hls_weights(hls_model)

    names.reverse()
    weights.reverse()
                      
    colors = cm.Blues(np.linspace(0, 1, len(names)))
    
    bplot = ax[0][1].boxplot(weights, vert=False, medianprops=dict(linestyle='-', color='k'),
                             showfliers=False, patch_artist=True, labels=names)
    fig.canvas.draw()
    
    ticks = {tick.get_text():tick.get_position()[1] for tick in ax[0][1].get_yticklabels()}
    
    for i, label in enumerate(weight_precisions['layer']):
        if label in ticks.keys():
            low = 2**weight_precisions['low'][i]
            high = 2**weight_precisions['high'][i]
            y = ticks[label]
            rectangle = Rectangle((low, y - 0.4), high - low, 0.8, fill=True, color='grey', alpha=0.2)
            ax[0][1].add_patch(rectangle)

    ax[0][1].set_xscale('log', base=2)
    ax[0][1].set_title('Distribution of weights after optimization')
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    names, activations = keras_activations(model, x)
    
    activations.reverse()
    names.reverse()
    
    colors = cm.Blues(np.linspace(0, 1, len(names)))
    
    bplot = ax[1][0].boxplot(activations, vert=False, medianprops=dict(linestyle='-', color='k'),
                             showfliers=False, patch_artist=True, labels=names)
    
    fig.canvas.draw()
    
    ticks = {tick.get_text():tick.get_position()[1] for tick in ax[1][0].get_yticklabels()}
    
    for i, label in enumerate(activation_precisions['layer']):
        if label in ticks.keys():
            low = 2**activation_precisions['low'][i]
            high = 2**activation_precisions['high'][i]
            y = ticks[label]
            rectangle = Rectangle((low, y - 0.4), high - low, 0.8, fill=True, color='grey', alpha=0.2)
            ax[1][0].add_patch(rectangle)

    ax[1][0].set_xscale('log', base=2)
    ax[1][0].set_title('Distribution of activations after optimization')
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
    names, activations = hls_activations(hls_model, x)
    
    activations.reverse()
    names.reverse()
    
    colors = cm.Blues(np.linspace(0, 1, len(names)))
    
    bplot = ax[1][1].boxplot(activations, vert=False, medianprops=dict(linestyle='-', color='k'),
                             showfliers=False, patch_artist=True, labels=names)
    
    fig.canvas.draw()
    
    ticks = {tick.get_text():tick.get_position()[1] for tick in ax[1][1].get_yticklabels()}
    
    for i, label in enumerate(activation_precisions['layer']):
        if label in ticks.keys():
            low = 2**activation_precisions['low'][i]
            high = 2**activation_precisions['high'][i]
            y = ticks[label]
            rectangle = Rectangle((low, y - 0.4), high - low, 0.8, fill=True, color='grey', alpha=0.2)
            ax[1][1].add_patch(rectangle)

    ax[1][1].set_xscale('log', base=2)
    ax[1][1].set_title('Distribution of activations after optimization')
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    return fig

def training(history, metrics, validate=True, title=None, xlabel='epoch', ylabel=None, scale=None):
    """ Visualize the training history by a given metric (accuracy, loss, etc) """
    labels = []
    fig = plt.figure()
    ax = fig.add_subplot()
    if title is None: title = metrics[0]
    if ylabel is None: ylabel = metrics[0]
    
    for metric in metrics:
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(history[metric], linestyle='-', color=color)
        labels.append('train_' + metric)
        if validate:
            ax.plot(history['val_' + metric], linestyle='--', color=color)
            labels.append('val_' + metric)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if scale is not None: plt.yscale(scale)
    ax.legend(labels, loc='best')
    return fig

def trainingOverlay(histories, labels, metric, title=None, xlabel='epoch', ylabel=None, scale=None):
    """ Overlay multiple training histories to compare model convergence """
    fig = plt.figure()
    ax = fig.add_subplot()
    if title is None: title = metric
    if ylabel is None: ylabel = metric
    colors = ['#733635ff', '#355373ff', '#357355ff'] # Theme colors
    for i, history in enumerate(histories):
        color = colors[int(i/2)]
        if i % 2 == 0:
            linestyle = '-'
        else:
            linestyle = '--'
        ax.plot(history[metric], linestyle=linestyle, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if scale is not None: plt.yscale(scale)
    ax.legend(labels, loc='best')
    return fig

def roc(preds, targets, labels, title='ROC', xlabel='Rejection Rate', ylabel='Identification Rate'):
    """ Receiver operating characteristic curve for visualizing models' classification performances """
    fig = plt.figure()
    ax = fig.add_subplot()
    
    fps = {}
    tps = {}
    scores = {}
    
    for i in range(len(labels)):
        fps[labels[i]], tps[labels[i]], _ = roc_curve(targets[i], preds[i])
        scores[labels[i]] = roc_auc_score(targets[i], preds[i])
        
    for i, label in enumerate(labels):
        ax.plot(1-fps[label], tps[label])
        labels[i] = labels[i] + f' AUC: {scores[label]:.3f}'
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(labels, loc='best')
    return fig

def efficiency(preds, targets, labels, title='Efficiency', xlabel='Efficiency', ylabel='Rejection Rate'):
    """ Adaptation of the ROC for finer detail comparing models' classification performances """
    fig = plt.figure()
    ax = fig.add_subplot()
    
    fps = {}
    tps = {}
    
    for i in range(len(labels)):
        fps[labels[i]], tps[labels[i]], _ = roc_curve(targets[i], preds[i])
   
    with np.errstate(divide='ignore'):
        for label in labels:
            ax.semilogy(tps[label], 1/fps[label])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(labels, loc='best')
    return fig

def regResponse(pred, target, stat=['median'], bins=None, title='Regression Response', 
                xlabel='True Cluster Energy', ylabel='Predicted Energy / True Energy'):
    """ Regression response curve for visualizing a model's regression performance """
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot()

    with np.errstate(divide='ignore'):
        x = target
        y = np.nan_to_num(pred / target, 0.0)

    try: 
        assert len(bins) == 2
        xbin = bins[0]
        ybin = bins[1]
    except:
        xbin = [10**exp for exp in np.arange(-0.8, 3.1, 0.1)]
        ybin = np.arange(0.5, 2.1, 0.025)

    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin) - 1)]
    
    cmap = 'viridis'
    hh = ax.hist2d(x, y, bins=[xbin, ybin], cmap=cmap, norm=LogNorm(), density=True, zorder=-1)
    ax.plot([0.1, 1000], [1, 1], linestyle='--', color='black')
    for s in stat:
        if s == 'stdmean':
            upper = stats.binned_statistic(x, y, bins=xbin, statistic='mean').statistic + \
            np.abs(stats.binned_statistic(x, y, bins=xbin, statistic='std').statistic)
            lower = stats.binned_statistic(x, y, bins=xbin, statistic='mean').statistic - \
            np.abs(stats.binned_statistic(x, y, bins=xbin, statistic='std').statistic)
            ax.fill_between(xcenter, lower, upper, color='black', alpha=0.2)
        else:      
            ps = stats.binned_statistic(x, y, bins=xbin, statistic=s).statistic
            #color='#8cc9caff'
            color='red'
            ax.plot(xcenter, ps, color=color)
    ax.set_xscale('log')
    ax.set_ylim(0.5, 2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(hh[3], ax=ax, label='Fraction of Clusters')
    return fig

def regResponseOverlay(preds, targets, labels, stat=['median'], bins=None, title='Regression Response Comparison',
                       xlabel='True Cluster Energy', ylabel='Predicted Energy / True Energy'):
    """ Overlay of regression responses for comparing regression performance between models """
    fig = plt.figure()
    ax = fig.add_subplot()
    colors = ['#733635ff', '#355373ff', '#357355ff'] # Theme colors
    
    x = {}
    y = {}
    
    with np.errstate(divide='ignore'):
        for i in range(len(labels)):
            x[labels[i]] = targets[i]
            y[labels[i]] = np.nan_to_num(preds[i] / targets[i], posinf=1000.0, neginf=-1000.0)
    try: 
        assert len(bins) == 2
        xbin = bins[0]
        ybin = bins[1]
    except:
        xbin = [10**exp for exp in np.arange(-1.0, 3.1, 0.08)]
        
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin) - 1)]
    
    for s in stat:
        iq = 0
        ic = 0
        for label in labels:
            if 'quantized' in label.lower():
                linestyle = '--'
                color = colors[iq]
                iq = iq + 1
            else:
                linestyle = '-'
                color = colors[ic]
                ic = ic + 1
            if s == 'stdmean':
                upper = stats.binned_statistic(x[label], y[label], bins=xbin, statistic='mean').statistic + \
                np.abs(stats.binned_statistic(x[label], y[label], bins=xbin, statistic='std').statistic)
                lower = stats.binned_statistic(x[label], y[label], bins=xbin, statistic='mean').statistic - \
                np.abs(stats.binned_statistic(x[label], y[label], bins=xbin, statistic='std').statistic)
                ax.fill_between(xcenter, lower, upper, color=color, alpha=0.2)
            else:      
                ps = stats.binned_statistic(x[label], y[label], bins=xbin, statistic=s).statistic
                ax.plot(xcenter, ps, linestyle=linestyle, color=color)

    ax.plot([0.1, 1000], [1, 1], linestyle='--', color='black', zorder=0)

    ax.set_xscale('log')
    ax.set_ylim(0.8, 2)
    ax.set_xlim(0.1, 1000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(labels, loc='best')
    return fig 
    
    