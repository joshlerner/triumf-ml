import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.stats as stats

class Plotter:
    """ """
    def __init__(self, function, **kwargs):
        """ """
        self.fig = function(**kwargs)
        
    def show(self):
        self.fig.show()
        
    def save(self, outpath, **kwargs):
        self.fig.savefig(outpath, **kwargs) 

def training(history, metric, validate=True, title=None, xlabel='epoch', ylabel=None, scale=None):
    """ """
    labels = []
    fig = plt.figure()
    ax = fig.add_subplot()
    if title is None: title = metric
    if ylabel is None: ylabel = metric
    ax.plot(history[metric])
    labels.append('train')
    if validate:
        ax.plot(history['val_' + metric])
        labels.append('val')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if scale is not None: plt.yscale(scale)
    ax.legend(labels, loc='best')
    return fig

def roc(pred, target, label='', title='ROC', xlabel='Rejection Rate', ylabel='Identification Rate'):
    labels = []
    labels.append(label)
    fig = plt.figure()
    ax = fig.add_subplot()
    fp, tp, threshs = roc_curve(target, pred)
    score = roc_auc_score(target, pred)
    ax.plot(1-fp, tp)
    labels[0] = labels[0] + f' AUC: {score:.3f}'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(labels, loc='best')
    return fig

def efficiency(pred, target, label=None, title='Classification Efficiency', xlabel='Efficiency', ylabel='Rejection Rate'):
    fig = plt.figure()
    ax = fig.add_subplot()
    fp, tp, threshs = roc_curve(target, pred)
    with np.errstate(divide='ignore'):
        ax.semilogy(tp, 1/fp)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if label is not None: ax.legend(label, loc='best')
    return fig

def regResponse(pred, target, stat=['median'], bins=None, title='Regression Response', 
                xlabel='Cluster Calib Hits', ylabel='Cluster Energy / Calib Hits'):
    fig = plt.figure()
    ax = fig.add_subplot()
    with np.errstate(divide='ignore'):
        x = target
        y = np.nan_to_num(pred / target, 0.0)
    try: 
        assert len(bins) == 2
        xbin = bns[0]
        ybin = bins[1]
    except:
        xbin = [10**exp for exp in np.arange(-0.9, 3.1, 0.1)]
        ybin = np.arange(0.1, 3.1, 0.1)
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin) - 1)]
    hh = ax.hist2d(x, y, bins=[xbin, ybin], cmap='gist_earth_r', norm=LogNorm(), zorder=-1)
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
            ax.plot(xcenter, ps, color='red')
    ax.set_xscale('log')
    ax.set_ylim(0, 3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(hh[3], ax=ax)
    return fig

def regResponseOverlay(xdict, ydict, stat='median',
                       xlabel='Cluster Calib Hits',
                       ylabel='Cluster Energy / Calib Hits',
                       bins=None):
    """ """
    fig = plt.figure()
    ax = fig.add_subplot()
    
    try: 
        assert len(bins) == 2
        xbin = bins[0]
        ybin = bins[1]
    except:
        xbin = [10**exp for exp in np.arange(-1.0, 3.1, 0.1)]
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin) - 1)]
    profile_stats = {label:stats.binned_statistic(xdict[label], ydict[label], bins=xbin, statistics=stat).statistic for label in xdict}
    labels = xdict.keys()
    ax.plot([0.1, 1000], [1, 1], linestyle='--', color='black')
    for label in labels:
        ax.plot(xcenter, profile_stats[label])
    ax.set_xscale('log')
    ax.set_ylim(0, 3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(labels, loc='best')
    return fig 
    
    