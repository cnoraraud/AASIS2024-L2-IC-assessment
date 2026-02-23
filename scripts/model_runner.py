import torch
import chart_reader as cr
import dataframe_sourcer as dfs
import torch_models as tm
import torch.nn as nn
import numpy as np
import pandas as pd
import io_tools as iot
from collections import Counter
from torch.utils.data import DataLoader
from datetime import datetime
from torcheval.metrics.functional import multiclass_accuracy
import os
import data_logger as dl
import keywords_recipes as kwr
import torch_wrapper as tw
import pickle

RS_EVAL = 4
RS_TEST = 10
FRAC_EVAL = 0.1
FRAC_TEST = 0.2

def dataset_load(speakers, recipe):
    complete_dataset = cr.ChartDataset(recipe)
    n, e = complete_dataset.get_embed_dims()
    datasets = {}
    datasets["train_dataset"] = cr.ChartDataset(recipe, allowed_speakers=speakers["train_speakers"], n=n, e=e)
    datasets["test_dataset"] = cr.ChartDataset(recipe, allowed_speakers=speakers["test_speakers"], n=n, e=e)
    datasets["mixed_dataset"] = cr.ChartDataset(recipe, allowed_speakers=speakers["train_speakers"]+speakers["test_speakers"], n=n, e=e)
    datasets["eval_dataset"] = cr.ChartDataset(recipe, allowed_speakers=speakers["eval_speakers"], n=n, e=e)
    datasets["train_distribution"] = cr.get_dataset_distribution(datasets["train_dataset"])
    datasets["test_distribution"] = cr.get_dataset_distribution(datasets["test_dataset"])
    return n, e, datasets

def create_dataloaders(datasets, recipe, collate_fn, shuffle=True):
    batch_size = recipe.get(kwr.batch_size)
    dataloaders = {}
    dataloaders["train_dataloader"] = DataLoader(datasets["train_dataset"], batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    dataloaders["test_dataloader"] = DataLoader(datasets["test_dataset"], batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    dataloaders["mixed_dataloader"] = DataLoader(datasets["mixed_dataset"], batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    dataloaders["eval_dataloader"] = DataLoader(datasets["eval_dataset"], batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    dataloaders["train_batches"] = len(dataloaders["train_dataloader"])
    dataloaders["test_batches"] = len(dataloaders["test_dataloader"])
    return dataloaders

def prepare_data(speakers, recipe):
    n, e, datasets = dataset_load(speakers, recipe)
    dataloaders = create_dataloaders(datasets, recipe, cr.collate_stack)
    
    data = {**speakers, **datasets, **dataloaders}
    return n, e, data

def prepare_model(n, e, recipe, weights=None):
    embedder = recipe.get(kwr.embedder)
    model = tm.SimpleArchitecture([0,e,n], recipe, embedder)
    model = model.to(device=recipe.get(kwr.device), dtype=recipe.get(kwr.dtype))
    loss_fn_class = recipe.get(kwr.loss_fn_class)

    if recipe.get(kwr.use_weights):
        loss_fn = loss_fn_class(weight=weights)
    else:
        loss_fn = loss_fn_class()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.get(kwr.learning_rate), weight_decay=recipe.get(kwr.weight_decay))
    
    return model, loss_fn, optimizer

def model_loss(x, y, model, loss_fn, recipe):
    loss_predictions = model(x)
    targets = y
    targets = targets.nan_to_num(nan=targets.nanmean())

    predictions = loss_predictions
    loss_targets = targets
    confidences = torch.zeros_like(targets)
    if recipe.get(kwr.classes):
        loss_targets = (targets.round().to(dtype=torch.long) - 1)
    if  recipe.get(kwr.classification_type)=="categorical":
        max_vals, max_args = torch.max(loss_predictions, -1)
        predictions = max_args + 1
        if not recipe.get(kwr.skip_logsoftmax):
            confidences = torch.exp(max_vals)
        else:
            confidences = max_vals
    if  recipe.get(kwr.classification_type)=="corn":
        predictions = loss_fn.inference(loss_predictions.flatten(0,1)) + 1
        # confidence measurable?
    if recipe.get(kwr.classes) or recipe.get(kwr.y_n) != 1:
        loss_predictions = loss_predictions.flatten(0,1)
        loss_targets = loss_targets.flatten(0,1)
    loss = loss_fn(loss_predictions, loss_targets)

    return loss, class_scores(predictions), class_scores(targets), confidences

def get_loss_score(loss):
    return loss.item()

def get_accuracy_score(predictions, targets):
    return multiclass_accuracy(predictions, targets).item()

def get_confidence_score(confidences, classes=False):
    if not classes:
        return None
    return torch.nanmean(confidences).item()

def train_model_epoch(set_id, name, epoch, model, train_dataloader, loss_fn, optimizer, recipe, report_every=5):
    loss_scores = []
    accuracies = []
    confidence_scores = []
    model.train()
    N = len(train_dataloader)
    rows = []
    for batch, (x, y, length) in enumerate(train_dataloader):
        loss, predictions, targets, confidences = model_loss(x, y, model, loss_fn, recipe)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_score = get_loss_score(loss)
        accuracy = get_accuracy_score(predictions, targets)
        confidence_score = get_confidence_score(confidences, classes=recipe.get(kwr.classes))

        loss_scores.append(loss_score)
        accuracies.append(accuracy)
        confidence_scores.append(confidence_score)

        target_classes = y_to_class_counts(targets)
        predicition_classes = y_to_class_counts(predictions)
        row = report_batch(set_id, name, batch, epoch, model, loss_score, accuracy, confidence_score, target_classes, predicition_classes, report=False)
        rows.append(row)
        num = batch + 1
        if num % report_every == 0 or num == N:
            average_loss = np.nanmean(loss_scores[-report_every:]).item()
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"[{num}/{N}] - train: {loss_score:.3f}, average of previous {report_every}: {average_loss:.3f}", 2)
            report_batches(set_id, name, model, rows)

    return loss_scores, accuracies, confidence_scores

def test_model(epoch, name, model, test_dataloader, loss_fn, recipe):
    loss_scores = []
    accuracies = []
    confidence_scores = []
    model.eval()
    with torch.no_grad():
        for batch, (x, y, length) in enumerate(test_dataloader):
            loss, predictions, targets, confidences = model_loss(x, y, model, loss_fn, recipe)
            
            loss_score = get_loss_score(loss)
            accuracy = get_accuracy_score(predictions, targets)
            confidence_score = get_confidence_score(confidences, classes=recipe.get(kwr.classes))

            loss_scores.append(loss_score)
            accuracies.append(accuracy)
            confidence_scores.append(confidence_score)
    
    return loss_scores, accuracies, confidence_scores

def get_model_name(model):
    name = model.__class__.__name__
    model_id = hex(id(model))
    return f"{name}_{model_id}"

def class_scores(y):
    return y.flatten().clamp(1, 6).round().int()

def y_to_class_counts(y):
    values = class_scores(y)
    return torch.bincount(values, minlength=6)[-6:].tolist()

def append_to_csv(rows, set_id, file_name, report_to=0):
    df = pd.DataFrame(rows)
    if report_to == -1 or report_to==1:
        full_path = iot.train_reports_path() / file_name
        df.to_csv(full_path, mode="a", sep="\t", index=False, header=(not os.path.isfile(full_path)))
    if report_to == 0 or report_to==1:
        dir_path = iot.train_reports_path() / set_id
        iot.create_missing_folder_recursive(dir_path)
        full_path = dir_path / file_name
        df.to_csv(full_path, mode="a", sep="\t", index=False, header=(not os.path.isfile(full_path)))
        
def m(x, n=1, type="whole"):
    x = np.atleast_1d(x).astype(np.float64)
    if type == "whole":
        selection = x
    if type == "first":
        selection = x[:n]
    if type == "last":
        neg_n = -n
        selection = x[neg_n:]
    return np.nanmean(selection).item()

def report_batch(set_id, name, batch, epoch, model, loss, accuracy, confidence_score, target_classes, pred_classes, report=False):
    model_id = hex(id(model))
    row = {
        "name": name,
        "model_id": model_id,
        "time": datetime.now(),
        "epoch": epoch,
        "batch": batch,
        "loss": loss,
        "acc": accuracy,
        "conf": confidence_score,
        "A1": target_classes[0], "A2": target_classes[1],
        "B1": target_classes[2], "B2": target_classes[3],
        "C1": target_classes[4], "C2": target_classes[5],
        "A1p": pred_classes[0], "A2p": pred_classes[1],
        "B1p": pred_classes[2], "B2p": pred_classes[3],
        "C1p": pred_classes[4], "C2p": pred_classes[5],
    }
    if report:
        append_to_csv([row], set_id, file_name=f"batch_report_{name}.csv")
    return row    

def report_batches(set_id, name, model, rows):
    append_to_csv(rows, set_id, file_name=f"batch_report_{name}.csv")
    rows.clear()

def report_epoch(set_id, name, epoch, model, train_losses, test_losses, epoch_train_accuracy, epoch_test_accuracy, epoch_train_confidence, epoch_test_confidence, report=True):
    model_id = hex(id(model))
    row = {
        "name": name,
        "model_id": model_id,
        "time": datetime.now(),
        "epoch": epoch,
        "starting_loss": m(train_losses,type="first"),
        "ending_loss": m(train_losses,type="last"),
        "train_loss": m(train_losses),
        "test_loss": m(test_losses),
        "train_acc": epoch_train_accuracy,
        "test_acc": epoch_test_accuracy,
        "train_conf": epoch_train_confidence,
        "test_conf": epoch_test_confidence,
    }
    if report:
        append_to_csv([row], set_id, file_name=f"epoch_report_{name}.csv")

def report_model_development(model, set_id, recipe, results, datasplit, data, report=True):
    model_id = hex(id(model))
    total_params, trainable_params = tw.get_param_counts(model)
    train_distribution = data["train_distribution"]
    test_distribution = data["test_distribution"]
    row = {
        "name": recipe.get(kwr.model_name) + recipe.get(kwr.recipe_name),
        "model_id": model_id,
        "set_id": set_id,
        "started": results["started"],
        "ended": results["ended"],
        "dataset": recipe.get(kwr.label_file),
        "eval_split_seed": datasplit.eval_split_seed,
        "test_split_seed": datasplit.test_split_seed,
        "eval_split_ratio": datasplit.eval_split_ratio,
        "test_split_ratio": datasplit.test_split_ratio,
        "learning_rate": recipe.get(kwr.learning_rate),
        "weight_decay": recipe.get(kwr.weight_decay),
        "batch_size": recipe.get(kwr.batch_size),
        "train_batches": data["train_batches"],
        "context_length": recipe.get(kwr.context),
        "epochs": results.get("last_epoch"),
        "final_train_loss": m(results.get("train_losses"),type="last"),
        "final_test_loss": m(results.get("test_losses"),type="last"),
        "final_train_acc": m(results.get("train_accuracies"),type="last"),
        "final_test_acc": m(results.get("test_accuracies"),type="last"),
        "final_train_conf": m(results.get("train_confidences"),type="last"),
        "final_test_conf": m(results.get("test_confidences"),type="last"),
        "device": recipe.get(kwr.device),
        "dtype": recipe.get(kwr.dtype),
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "A1_train": train_distribution[0], "A2_train": train_distribution[1],
        "B1_train": train_distribution[2], "B2_train": train_distribution[3],
        "C1_train": train_distribution[4], "C2_train": train_distribution[5],
        "A1_test": test_distribution[0], "A2_test": test_distribution[1],
        "B1_test": test_distribution[2], "B2_test": test_distribution[3],
        "C1_test": test_distribution[4], "C2_test": test_distribution[5],
    }
    if report:
        append_to_csv([row], set_id, file_name=f"dev_reports.csv", report_to=1)

def develop_model(set_id, name, model, train_dataloader, test_dataloader, loss_fn, optimizer, recipe):
    dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, "Starting model development", 0)
    
    dev_train_losses = []
    dev_test_losses = []
    dev_train_accuracies = []
    dev_test_accuracies = []
    dev_train_confidences = []
    dev_test_confidences = []

    convergence_limit = 3
    convergence_count = 0
    last_epoch = 0
    for epoch in range(recipe.get(kwr.epochs)):
        last_epoch = epoch
        ep_train_losses, ep_train_accuracies, ep_train_confidences = train_model_epoch(set_id, name, epoch, model, train_dataloader, loss_fn, optimizer, recipe)
        ep_test_losses, ep_test_accuracies, ep_test_confidences = test_model(epoch, name, model, test_dataloader, loss_fn, recipe)

        ep_train_loss = np.nanmean(ep_train_losses).item()
        ep_test_loss = np.nanmean(ep_test_losses).item()
        ep_train_accuracy = np.nanmean(ep_train_accuracies).item()
        ep_test_accuracy = np.nanmean(ep_test_accuracies).item()
        ep_train_confidence = None
        ep_test_confidence = None
        if recipe.get(kwr.classes):
            ep_train_confidence = np.nanmean(ep_train_confidences).item()
            ep_test_confidence = np.nanmean(ep_test_confidences).item()

        dev_train_losses.append(ep_train_loss)
        dev_test_losses.append(ep_test_loss)
        dev_train_accuracies.append(ep_train_accuracy)
        dev_test_accuracies.append(ep_test_accuracy)
        dev_train_confidences.append(ep_train_confidence)
        dev_test_confidences.append(ep_test_confidence)

        num = epoch + 1
        dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"[{num}/{recipe.get(kwr.epochs)}] - train: {ep_train_loss:.3f}, test: {ep_test_loss:.3f}", 1)
        report_epoch(set_id, name, epoch, model, ep_train_losses, ep_test_losses, ep_train_accuracy, ep_test_accuracy, ep_train_confidence, ep_test_confidence, report=True)

        store_model(set_id, name, model, optimizer, epoch, ep_train_losses[-1])
        
        # Early stopping
        if ep_train_accuracy >= 0.98:
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, "Early stop due to high train accuracy!")
            break
        #if epoch > 3 and ep_test_loss > 5:
        #    dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, "Early stop due to bad test loss!")
        #    break
        if epoch > 10 and (dev_train_losses[-1] > dev_train_losses[-2]):
            convergence_count += 1
            reference = dev_train_losses[-6]
            samples = dev_train_losses[-5:]
            mean_improvement = np.mean(reference - np.array(samples))
            mean_improvement_relative = mean_improvement/reference
            if mean_improvement_relative < 0.04:
                convergence_count += 1
        convergence_count = max(convergence_count - 1, 0)
        if convergence_count >= convergence_limit:
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, "Early stop due to convergence!")
            break
        

    dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, "Finished model development", 0)

    res = {
        "train_losses": dev_train_losses,
        "test_losses": dev_test_losses,
        "train_accuracies": dev_train_accuracies,
        "test_accuracies": dev_test_accuracies,
        "train_confidences": dev_train_confidences,
        "test_confidences": dev_test_confidences,
        "last_epoch": last_epoch    
    }
    return res

def store_model(set_id, name, model, optimizer, epoch=-1, loss=-1):
    dir_path = iot.torch_models_path() / set_id
    iot.create_missing_folder_recursive(dir_path)
    fullpath = dir_path / f"{name}.pt"
    torch.save({"epoch": epoch, "loss": loss, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, fullpath)

def get_set_id(name):
    pid = dl.get_process_id()
    timestamp = dl.get_timestamp(clean=True)
    return f"t{timestamp}_p{pid}_{name}"

def limit_df_by_channels(df, th=3):
    ch1 = Counter(df["channel_1"].value_counts().to_dict())
    ch2 = Counter(df["channel_2"].fillna("none").value_counts().to_dict())
    ch1c = Counter()
    for l in ch1.keys():
        val = ch1[l]
        if val >= th:
            ch1c[l] = ch1[l]
    ch2c = Counter()
    for l in ch2.keys():
        val = ch2[l]
        if val >= th:
            ch2c[l] = ch2[l]
    chac = ch1c + ch2c
    chacl  = list(chac.keys())
    final_df = df[(df["channel_1"].isin(chacl)) & (df["channel_2"].isin(chacl))]
    return final_df

def save_keyfields(kf, ver=""):
    file_path = iot.output_csvs_path() / iot.SELECTINFO_FOLDER
    iot.create_missing_folder_recursive(file_path)
    with open(file_path / f"keyfields{ver}.pickle", "wb") as f:
        pickle.dump(kf, f)
def load_keyfields(ver=""):
    kf = []
    file_path = iot.output_csvs_path() / iot.SELECTINFO_FOLDER
    with open(file_path / f"keyfields{ver}.pickle", "rb") as f:
        kf = pickle.load(f)
    return kf

class Datasplit(object):

    def __init__(self, tasks=["task5"], eval_split_seed = 4, test_split_seed = 10, eval_split_ratio = 0.2, test_split_ratio = 0.2, n = -1, fold_n=1):
        self.tasks = tasks
        self.eval_split_seed = eval_split_seed
        self.test_split_seed = test_split_seed
        self.eval_split_ratio = eval_split_ratio
        self.test_split_ratio = test_split_ratio
        self.seed_mult = 117
        self.k = 0
        self.n = n
        self.fold_n = fold_n
    
    def set_seed_mult(self, mult):
        self.seed_mult = mult

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def __getitem__(self, k):
        eval_split_seed = self.eval_split_seed
        test_split_seed = self.test_split_seed + self.seed_mult * k
        eval_split_ratio = self.eval_split_ratio
        test_split_ratio = self.test_split_ratio
        return dfs.get_all_speaker_splits(eval_split_seed, test_split_seed, eval_split_ratio, test_split_ratio, self.tasks, self.fold_n)

    def next(self):
        if self.k < self.n or self.n < 0:
            self.k += 1
            return self[self.k - 1]
        raise StopIteration()


def experiment_model(recipe):
    name = recipe.get(kwr.model_name) + recipe.get(kwr.recipe_name)
    set_id = get_set_id(name)
    
    iterations = recipe.get(kwr.iterations)
    dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"Experimenting with model. Iterations count: {iterations}", 0)
    datasplit = Datasplit()
    for iteration in range(iterations):
        speakers = next(datasplit)
        folds = speakers["folds"]
        for fold in range(folds):
            fold_speakers = {
                "eval_speakers": speakers["eval_speakers"],
                "train_speakers": speakers["train_speakers"][fold],
                "test_speakers": speakers["test_speakers"][fold],
            }
            n, e, data = prepare_data(fold_speakers, recipe)
            weights = torch.Tensor(cr.get_weights_from_distribution(data["train_distribution"], recipe.get(kwr.weight_type), recipe.get(kwr.weight_power))).to(device=recipe.get(kwr.device), dtype=recipe.get(kwr.dtype))

            started = datetime.now()
            model, loss_fn, optimizer = prepare_model(n, e, recipe, weights)
            
            if recipe.get(kwr.device)=="cuda":
                pass
                #model.compile() #TODO: Figure out
            model_name = get_model_name(model)
            train_speakers_joined = ", ".join(str(x) for x in data["train_speakers"])
            test_speakers_joined = ", ".join(str(x) for x in data["test_speakers"])
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"Training model {iteration + 1} out of {iterations} in set \'{set_id}\' with test split seed {speakers["test_split_seed"]}", 0)
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"Fold {fold + 1} out of {folds}")
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"Train distribution: {data["train_distribution"]}", 0)
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"Test distribution: {data["test_distribution"]}", 0)
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"Train speakers:\n{train_speakers_joined}", 0)
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"Test speakers:\n{test_speakers_joined}", 0)
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"{model_name}", 0)
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"{model}", 0)
            dl.write_to_manifest_log(dl.MODEL_TRAINING_TYPE, f"{optimizer}", 0)
            results = develop_model(set_id, name, model, data["train_dataloader"], data["test_dataloader"], loss_fn, optimizer, recipe)
            ended = datetime.now()
            results["started"] = started
            results["ended"] = ended

            report_model_development(model, set_id, recipe, results, datasplit, data, report=True)