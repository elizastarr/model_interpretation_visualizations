from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

from IPython.display import set_matplotlib_formats, display, HTML
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openml as oml
import os
from cycler import cycler
from pprint import pprint
import mglearn
import seaborn as sns
from tabulate import tabulate

set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['image.cmap'] = "viridis"
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['savefig.bbox'] = "tight"
#plt.rcParams['lines.linewidth'] = 1
plt.rcParams['legend.numpoints'] = 1
plt.rc('axes', prop_cycle=(cycler('color', mglearn.plot_helpers.cm_cycle.colors) +
                           cycler('linestyle', ['-', '--', ':',
                                                '-.', '--'])
                           )
       )

np.set_printoptions(precision=3, suppress=True)

pd.set_option("display.max_columns", 8)
pd.set_option('precision', 2)

np, mglearn

# Prints outputs in cells so that we don't have to write print() every time 
#InteractiveShell.ast_node_interactivity = "all"

# Matplotlib tweaks for presentations
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.max_open_warning"] = -1
plt.rcParams['font.size'] = 8; 
plt.rcParams['lines.linewidth'] = 0.5


# Presentations
from notebook.services.config import ConfigManager
cm = ConfigManager()
cm.update('livereveal', {'width': '95%', 'height': 786, 'scroll': True, 'theme': 'serif', 'transition': 'fade', 'overflow': 'visible', 'start_slideshow_at': 'selected'})

# Silence warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


def print_methods():
    return pd.DataFrame([['roc(y_test, y_pred, y_pred_proba)','binary class','plot'],
              ['prob_density_curve(y_test, y_pred_proba)','binary class','plot'],
              ['prc(y_test, y_pred)','binary class','plot'],
              ['c_matrix(y_test, y_pred)','binary & multi class','plot'],
              ['prediction_samples(X_test, y_test, clf, feat_names)','binary & multi class','DataFrame'],
              ['dataset_balance(y_train, y_test, y_pred)','binary & multi class','plot'],      
              ['prediction_plot(y_test, y_pred)','reg','plot'],
              ['residual_plot(y_test, y_pred)','reg','plot'],
              ['feat_importance(clf, feat_names=None)','linear & ensemble','plot'],
              ['binary_classification(y_test, y_pred, y_pred_proba)','binary class','various'],
              ['multiclass_classification(y_test, y_pred, y_pred_proba)','multi class','various'],
              ['regression(y_test, y_pred)','reg','various']],
             columns=['method signature','task','output'])

print(tabulate(print_methods(), headers='keys', tablefmt='psql'))

def roc(y_test, y_pred, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
    
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_test, y_pred))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    
    plt.legend(loc=4, fontsize='large')
    plt.title("ROC Curve")
    plt.show()

def prc(y_test, y_pred, y_pred_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:,1])

    plt.plot(recall, precision, label="PR curve (average = %.2f)" % average_precision_score(y_test, y_pred))
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Precision Recall Curve")
    plt.legend(loc="best",fontsize='large')
    plt.show()

def c_matrix(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    target_names = np.unique(y_test)
    scores_image = mglearn.tools.heatmap(
        confusion_matrix(y_test, y_pred), 
        xlabel='Predicted label',
        ylabel='True label', 
        xticklabels=target_names,
        yticklabels=target_names, 
        cmap=plt.cm.Blues, 
        fmt="%d")    
    plt.title("Confusion Matrix")
    plt.gca().invert_yaxis()
    plt.show()

    scores_image = mglearn.tools.heatmap(
        confusion_matrix(y_test, y_pred, normalize='true'), 
        xlabel='Predicted label',
        ylabel='True label', 
        xticklabels=target_names,
        yticklabels=target_names, 
        cmap=plt.cm.Blues)    
    plt.title("Normalized Confusion Matrix")
    plt.gca().invert_yaxis()
    plt.show()
    
def optimal_threshold(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]
    
def prob_density_curve(y_test, y_pred_proba):
    sns.set(rc={'figure.figsize':(8,5)})

    thresh = optimal_threshold(y_test, y_pred_proba)
    plt.axvline(x=thresh, label='Threshold=%0.2f' % thresh)

    y_test_proba = np.concatenate([y_test.reshape(y_test.size, 1), y_pred_proba[:,1].reshape(y_test.size, 1)], axis=1)
    label_0 = y_test_proba[np.where(y_test_proba[:,0]==0)]
    label_1 = y_test_proba[np.where(y_test_proba[:,0]==1)]

    sns.kdeplot(label_0[:,1], color = 'r', shade=True, shade_lowest=False, label='0')
    sns.kdeplot(label_1[:,1], color = 'g', shade=True, shade_lowest=False, label='1')
    plt.title("Probability Density")
    plt.ylabel("Kernel Density Estimate")
    plt.legend(fontsize='large')
    plt.show()

def prediction_plot(y_test, y_pred):
    plt.subplot(1, 2, 1)
    plt.gca().set_aspect("equal")
    plt.plot([10, 50], [10, 50], '--', c='k')
    plt.plot(y_test, y_pred, 'o', alpha=.5)
    plt.title("Predicted v. Actual")
    plt.ylabel("predicted")
    plt.xlabel("true")
    
def residual_plot(y_test, y_pred):
    plt.subplot(1, 2, 2)
    plt.gca().set_aspect("equal")
    plt.plot([10, 50], [0,0], '--', c='k')
    plt.plot(y_test, y_test - y_pred, 'o', alpha=.5)
    plt.title("Residuals")
    plt.xlabel("true")
    plt.ylabel("true - predicted")
    plt.tight_layout()

def feat_importance(clf, feat_names=None):
    # feature_importance_ is for ensembe models: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
    # coef_ is an attribute of linear models: https://scikit-learn.org/stable/modules/linear_model.html
    try:
        importance = clf.feature_importances_
    except AttributeError:
        importance = clf.coef_
    except:
        print("AttributeError: The model does not have an attribute named feature_importances_ or coef_.")

    if feat_names is None:
        feat_names = np.arange(importance.size)

    for name, value in zip(feat_names,importance):
        print('Feature: %s, Score: %.5f' % (name, value))
    
    feat_importances = pd.Series(importance, index=feat_names)
    feat_importances.sort_values().plot(kind='barh')
    plt.show()
    
def prediction_samples(X_test, y_test, clf, feat_names):
    
    class_labels = clf.classes_
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    
    # concatenate arrays
    concat = np.concatenate([y_test.reshape(y_test.size, 1), 
                                  y_pred.reshape(y_pred.size, 1),
                                  y_pred_proba,
                                  X_test], 
                                  axis=1)
    # separate correct classifications and missclassifications
    correct = concat[np.where(concat[:,0]==concat[:,1])]
    miss = concat[np.where(concat[:,0] != concat[:,1])]

    # Binary classification
    if(class_labels.size==2):
        concat_feat_names = ['Label', 'Prediction','Negative Prob.','Positive Prob.']+feat_names
        # separate true 0 from true 1
        correct_0 = correct[np.where(correct[:,0]==0)]
        correct_1 = correct[np.where(correct[:,0]==1)]
        miss_0 = miss[np.where(miss[:,0]==0)] # where correct label was 0
        miss_1 = miss[np.where(miss[:,0]==1)] # where correct label was 1

        # find 5 best and worst classifications
        k = 3
        pos_proba = 3 # row 3 holds probabilities for the positive class
        neg_proba = 2 # row 3 holds probabilities for the negative class

        results = []
        ### best correct ###
        if correct_0.shape[0] >= k:
            best_0 = correct_0[np.argpartition(correct_0[:,neg_proba], kth=k)[:k]]
            results.append(best_0)
        if correct_1.shape[0] >= k:
            best_1 = correct_1[np.argpartition(correct_1[:,pos_proba], kth=k)[:k]]
            results.append(best_1)

        ### worst missclassification ###
        if miss_0.shape[0] >= k:
            worst_0 = miss_0[np.argpartition(miss_0[:, pos_proba], kth=k)[:k]]
            results.append(worst_0)
        if miss_1.shape[0] >= k:
            worst_1 = miss_1[np.argpartition(miss_1[:, neg_proba], kth=k)[:k]]
            results.append(worst_1)

        return pd.DataFrame(np.vstack(results), columns=concat_feat_names)  
    
    else:
        probability_names = []
        for label in class_labels:
            probability_names.append(str(label)+" Prob.")
        concat_feat_names = ['Label', 'Prediction']+probability_names+feat_names
        k = 3
        results = []
        for i, label in enumerate(class_labels, start=0):
            correct_label = correct[np.where(correct[:,0]==label)]
            miss_label = miss[np.where(miss[:,0]==label)]
            
            label_proba = 3+1  # index of the probability values for this label
            if correct_label.shape[0] >= k:
                best_label = correct_label[np.argpartition(correct_label[:,label_proba], kth=k)[-k:]]
                results.append(best_label)
            if miss_label.shape[0] >= k:
                worst_label = miss_label[np.argpartition(miss_label[:,label_proba], kth=k)[:k]]
                results.append(worst_label)
        return pd.DataFrame(np.vstack(results), columns=concat_feat_names)

def dataset_balance(y_train, y_test, y_pred):
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)

    perc_train = (counts_train/y_train.size).tolist()
    perc_test = (counts_test/y_test.size).tolist()
    perc_pred = (counts_pred/y_pred.size).tolist()

    df = pd.DataFrame(perc_train+perc_test+perc_pred, columns = ["Percent"])
    df['Label'] = unique_train.tolist()+unique_test.tolist()+unique_pred.tolist()
    df['Dataset'] = ['Train' for label in unique_train]+ ['Test' for label in unique_test] + ['Predict' for label in unique_pred]

    pivot_df = df.pivot(index='Dataset', columns='Label', values='Percent')
    pivot_df.plot.bar(stacked=True, title='Label Percentages')
    plt.show()
    
def binary_classification(y_test, y_pred, y_pred_proba):
    roc(y_test, y_pred, y_pred_proba)
    prc(y_test, y_pred, y_pred_proba)
    prob_density_curve(y_test, y_pred_proba)
    c_matrix(y_test, y_pred)

def multiclass_classification(y_test, y_pred, y_pred_proba):
    c_matrix(y_test, y_pred)
    dataset_balance(y_train, y_test, y_pred)
    
def regression(y_test, y_pred):
    prediction_plot(y_test, y_pred)
    residual_plot(y_test, y_pred)
    
    