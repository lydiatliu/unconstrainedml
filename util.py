"""Utility functions for experiments on calibration, sufficiency and separation"""

import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from statsmodels.stats import proportion
import pylab


def data_transform(df):
    """Preprocesses categorical variables"""
    feature_cols = pd.get_dummies(df)
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
    return data


def score_range(p, q, df):
    """Computes the mean target variable in a score range"""
    gp = df[df['score'] > p]
    lq = gp[gp['score'] <= q]
    return np.mean(lq['target'])


def get_frequencies(results, num_buckets=10, ts=None):
    """computes frequencies of target variable for buckets"""
    if ts is None:
        ts = np.linspace(0, 1, num_buckets+1)
    else:
        num_buckets = len(ts)-1
    frequencies = []
    for i in range(num_buckets):
        frequencies.append(score_range(ts[i], ts[i+1], results))
    return frequencies


def mean_score(results):
    """Computes the mean score for positive and negative targets"""
    mean_scores = []
    sd = [] # standard deviation of the mean. sd/root(n)
    target_0 = results[results['target'] == 0]
    target_1 = results[results['target'] == 1]
    mean_scores.append(np.mean(target_0['score']))
    mean_scores.append(np.mean(target_1['score']))
    sd.append(np.std(target_0['score'])/np.sqrt(len(target_0)))
    sd.append(np.std(target_1['score'])/np.sqrt(len(target_1)))
    return mean_scores, sd


def sep_err_feature(res, feature, value_list=None):
    """Computes the separation gap"""
    results = res.copy()
    if value_list is None:
        value_list = results[feature].unique()
    else:
        results = results[results[feature].isin(value_list)]
    results['feature_score_mean'] = 0
    results['score_mean'] = 0
    feature_score_means = []
    for value in value_list:
        # mean score for each group
        feature_score_mean, _ = mean_score(results[results[feature] == value])
        feature_score_means.append(feature_score_mean)

    score_mean, _ = mean_score(results)
    temp = [0, 1]

    for i in range(2):
        results.loc[results['target'] == temp[i], 'score_mean'] = score_mean[i]
        for j, value in enumerate(value_list):
            results.loc[(results['target'] == temp[i]) & (results[feature] == value),
                        'feature_score_mean'] = feature_score_means[j][i]

    results['feature_sep_dev'] = results['feature_score_mean'] - results['score_mean']
    results['feature_sep_dev'] = results['feature_sep_dev'].abs()
    return results['feature_sep_dev'].mean()


def group_calib_feature(res, feature, value_list=None, num_buckets=10, use_quantiles=False):
    """Computes the sufficiency gap"""
    results = res.copy()
    if value_list is None:
        value_list = results[feature].unique()
    else:
        results = results[results[feature].isin(value_list)]
    results['feature_score_mean'] = 0
    results['score_mean'] = 0
    feature_score_means = []
    if use_quantiles:
        temp = np.linspace(0, 1, num_buckets + 1)
        ts = np.quantile(results['score'], temp)  # quantile buckets
    else:
        ts = np.linspace(0, 1, num_buckets + 1)
    for value in value_list:
        feature_score_means.append(get_frequencies(results[results[feature] == value], ts=ts))

    score_mean = get_frequencies(results, ts=ts)

    for i in range(num_buckets):
        results.loc[results['score'].between(ts[i], ts[i + 1]), 'score_mean'] = score_mean[i]
        for j, value in enumerate(value_list):
            results.loc[(results['score'].between(ts[i], ts[i + 1])) & (results[feature] == value),
                        'feature_score_mean'] = feature_score_means[j][i]

    results['feature_calib_dev'] = results['feature_score_mean'] - results['score_mean']
    results['feature_calib_dev'] = results['feature_calib_dev'].abs()
    return results['feature_calib_dev'].mean()


def plot_calib(res, feature, value_list=None, num_buckets=10, num_to_plot=2, string='student',
               title_string='', legend=None, use_quantiles=False):
    """Makes calibration plot"""
    results = res.copy()
    if value_list is None:
        tem = res[feature].value_counts()
        tem = list(tem.index)
        value_list = tem[:num_to_plot]
    else:
        results = results[results[feature].isin(value_list)]

    pylab.rc('font', size='13')
    pylab.rc('axes', labelsize='large')
    pylab.rc('lines', linewidth=3)

    pylab.figure(2, figsize=(5.2, 5.2))
    pylab.title(title_string)
    deciles = range(1, num_buckets + 1)

    if use_quantiles:
        temp = np.linspace(0, 1, num_buckets + 1)
        ts = np.quantile(results['score'], temp)  # quantile buckets
    else:
        ts = np.linspace(0, 1, num_buckets + 1)

    for i in range(len(value_list)):
        value = value_list[i]
        num = get_frequencies2(results[results[feature] == value], num_buckets=num_buckets, ts=ts)
        plot_confidence(deciles, num[0], num[1], value)

    if legend is None:
        pylab.legend()
    else:
        pylab.legend(legend)
    pylab.ylabel('Rate of positive outcomes')
    pylab.xlabel('Score decile')
    pylab.ylim([0, 1])
    pylab.tight_layout()
    pylab.savefig('../figures/%s.svg' % string.replace(' ','-'))
    pylab.savefig('../figures/%s.pdf' % string.replace(' ','-'))
    pylab.show()
    pylab.close()


def score_range2(p, q, df):
    """Returns number of success and observations for given bucket."""
    gp = df[df['score'] > p]
    lq = gp[gp['score'] <= q]
    if len(lq['target']) > 1:
        return [sum(lq['target']), len(lq['target'])]
    else:
        return [np.nan, np.nan]


def get_frequencies2(marginals, num_buckets=10, ts=None):
    """Returns list of lists. The first list is successes
    for all buckets, the second one is total counts."""
    if ts is None:
        ts = np.linspace(0, 1, num_buckets + 1)
    else:
        num_buckets = len(ts) - 1
    frequencies = [[], []]
    for i in range(num_buckets):
        b = score_range2(ts[i], ts[i + 1], marginals)
        frequencies[0].append(b[0])
        frequencies[1].append(b[1])
    return frequencies


def plot_confidence(xs, n_succs, n_obss, label, linestyle=None,
                    color=None, confidence=0.95):
    """Plot a graph with confidence intervals where
       each x corresponds to a binomial random variable in which
       n_obs observations led to n_succ successes"""
    n_succs, n_obss = np.array(n_succs), np.array(n_obss)
    conf_lbs, conf_ubs = proportion.proportion_confint(n_succs, n_obss, alpha=1 - confidence)
    pylab.fill_between(xs, conf_ubs, conf_lbs, alpha=.2, color=color)
    ys = n_succs.astype(float) / n_obss
    if linestyle is None:
        pylab.plot(xs, ys, '.-', label=label, linestyle=linestyle, color=color)
    else:
        pylab.plot(xs, ys, '.-', label=label)
