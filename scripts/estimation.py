import pandas as pd
import numpy as np
import numpy_wrapper as npw
from sklearn.metrics import root_mean_squared_error, accuracy_score, confusion_matrix
import sklearn.base as sklb
import dataframe_reader as dfr
import data_displayer as dd


def create_info(name, c, a):
    return {"name": name, "c": c, "a": a, "preds": [], "targets": []}


def add_to_info(info, preds, targets):
    info["preds"] += preds
    info["targets"] += targets


def info_rmse(info):
    return root_mean_squared_error(info["targets"], info["preds"])


def info_acc(info):
    return accuracy_score(info["targets"], info["preds"])


def info_acc_cheat(info):
    target_classes = dfr.num_to_cefr(pd.Series(np.array(info["targets"]))).values
    pred_classes = dfr.num_to_cefr(pd.Series(np.array(info["preds"]))).values
    return accuracy_score(target_classes, pred_classes)


def inf_conf(info):
    return confusion_matrix(info["targets"], info["preds"], labels=dfr.CEFR_ORDER)


def scatter_plot_info(info, extra_title=None, fig=None, ax=None):
    title = "Regression"
    if npw.is_string(extra_title):
        title = f"{title} {extra_title}"
    dd.s(
        info["targets"],
        info["preds"],
        title=title,
        width=5,
        height=5,
        xlim=[1, 6],
        ylim=[1, 6],
        y_label="Targets",
        x_label=info["name"],
        fig=fig,
        ax=ax,
    )


def confusion_plot_info(info, extra_title=None, fig=None, ax=None):
    title = "Classification"
    if npw.is_string(extra_title):
        title = f"{title} {extra_title}"
    dd.r(
        inf_conf(info),
        title,
        dfr.CEFR_ORDER,
        automate_colorscale=False,
        width=5,
        height=5,
        annotate=True,
        normalize="x",
        y_label="Targets",
        x_label=info["name"],
        fig=fig,
        ax=ax,
    )


def y_col_to_title(y_col):
    parts = y_col.replace("y_", "").split("_")
    for i in range(len(parts)):
        parts[i] = parts[i].capitalize()
    return " ".join(parts)


def do_one_fold(
    test_df,
    train_df,
    y_col,
    x_columns,
    dummy_constructor,
    model_constructor,
    imputer_constructor,
    scaler_constructor,
    decomposition_constructor,
    train_info,
    test_info,
    dummy_info,
    classification,
):
    test_y = test_df[y_col]
    test_x = test_df[x_columns]
    train_y = train_df[y_col]
    train_x = train_df[x_columns]

    if classification:
        test_y = dfr.num_to_cefr(test_y)
        train_y = dfr.num_to_cefr(train_y)

    dummy_model = sklb.clone(dummy_constructor)
    model = sklb.clone(model_constructor)
    pre_model_1 = sklb.clone(imputer_constructor)
    pre_model_2 = sklb.clone(scaler_constructor)

    # Pre-processing
    train_x_pp = pre_model_2.fit_transform(pre_model_1.fit_transform(train_x))
    test_x_pp = pre_model_2.transform(pre_model_1.transform(test_x))

    if not isinstance(decomposition_constructor, type(None)):
        pre_model_3 = sklb.clone(decomposition_constructor)
        train_x_pp = pre_model_3.fit_transform(train_x_pp)
        test_x_pp = pre_model_3.transform(test_x_pp)

    # Fitting
    model.fit(train_x_pp, train_y)
    dummy_model.fit(train_x_pp, train_y)

    # Prediction
    pred_train_pp = model.predict(train_x_pp)
    pred_dummy_pp = dummy_model.predict(test_x_pp)
    pred_test_pp = model.predict(test_x_pp)

    if not classification:
        pred_train_pp = np.clip(pred_train_pp, 1, 6)
        pred_test_pp = np.clip(pred_test_pp, 1, 6)
        pred_dummy_pp = np.clip(pred_dummy_pp, 1, 6)

    add_to_info(train_info, list(pred_train_pp), list(train_y))
    add_to_info(test_info, list(pred_test_pp), list(test_y))
    add_to_info(dummy_info, list(pred_dummy_pp), list(test_y))
