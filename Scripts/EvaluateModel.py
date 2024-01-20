import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
import shap
import matplotlib.pyplot as plt


def evaluate_model(x_train, y_train, model):
    cv_scores = pd.DataFrame(
        cross_validate(model, x_train, y_train, cv=5, scoring=['neg_root_mean_squared_error', 'max_error',
                                                               'neg_median_absolute_error'])).iloc[:, 2:]
    cv_scores_mean = cv_scores.apply(lambda x: -1 * x).mean()
    explainer = shap.Explainer(model.predict, x_train)
    shap_values = explainer(x_train)
    beeswarm_plot = shap.plots.beeswarm(shap_values)
    return cv_scores_mean, beeswarm_plot



