import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def training_curve(history, metric, 
                   validate=True, 
                   title=None, 
                   xlabel='epoch', 
                   ylabel=None, 
                   scale=None):
    """ """
    labels = []
    fig = plt.figure()
    ax = fig.add_subplot()
    if title is None: title = metric
    if ylabel is None: ylabel = metric
    ax.plot(history.history[metric])
    labels.append('train')
    if validate:
        ax.plot(history.history['val_' + metric])
        labels.append('val')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if scale is not None: plt.yscale(scale)
    ax.legend(labels, loc='best')
    fig.show()
    
def ROC(pred, target, title='ROC'):
    """ """
    fig = plt.figure()
    ax = fig.add_subplot()
    fp, tp, threshs = roc_curve(target[:, 0], pred[:, 0])
    score = roc_auc_score(target[:, 0], pred[:, 0])
    ax.plot(1-fp, tp)
    ax.set_xlabel('Rejection Rate')        
    ax.set_ylabel('Identification Rate')
    ax.set_title(title)
    ax.text(0, 0, f'AUC: {score:.3f}')
    fig.show()


    