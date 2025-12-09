import torch
import torch_layers as tl
import torch_models as tm
import torch_wrapper as tw

device = "device"
dtype = "dtype"
gen_recipe = {
    device: None,
    dtype: torch.float,
}

labels_dir = "labels_dir"
charts_dir = "charts_dir"
label_file = "label_file"
tasks = "tasks"
batch_size = "batch_size"
context = "context"
hop = "hop"
add_time = "add_time"
transpose_x = "transpose_x"
data_recipe = {
    label_file: "chart_labels_64_16.csv",
    tasks: ["task5"],
    batch_size: 128,
    context: 32,
    hop: 1,
    add_time: True,
    transpose_x: True,
}

iterations = "iterations"
epochs = "epochs"
learning_rate = "learning_rate"
weight_decay = "weight_decay"
weight_power = "weight_power"
weight_type = "weight_type"
learning_recipe = {
    iterations: 5,
    epochs: 50,
    learning_rate: 1e-3,
    weight_decay: 1e-5,
    weight_power: 1,
    weight_type: "p",
}


recipe_name = "recipe_name"
model_name = "model_name"
activation = "activation"
cnn_layer = "cnn_layer"
base_recipe = {
    recipe_name: "",
    activation: torch.nn.ReLU,
    cnn_layer: tl.CNNnD,
}

y_n = "y_n"
dropout_rate = "dropout_rate"
dims = "dims"
pool_p = "pool_p"
squeeze_r = "squeeze_r"
skip_norm = "skip_norm"
skip_posenc = "skip_pos"
skip_res = "skip_res"
skip_se = "skip_se"
skip_preembed = "skip_preembed"
pooling_kernel = "pooling_kernel"
hyperparameter_recipe = {
    y_n: 1,
    dropout_rate: 0,
    dims: 1,
    squeeze_r: 2,
    pool_p: 1.0,
    skip_norm: False,
    skip_posenc: False,
    skip_res: True,
    skip_se: True,
    skip_preembed: True,
    pooling_kernel: (1,4),
}

output_dim = "output_dim"
classes = "classes"
skip_logsoftmax = "skip_logsoftmax"
classification_type = "classification_type"
loss_fn_class = "loss_fn_class"
use_weights = "use_weights"

log_categorical_classification_recipe = {
    classes: True,
    use_weights: True,
    output_dim: 6,
    skip_logsoftmax: False,
    recipe_name: "Classifier",
    classification_type: "categorical",
    loss_fn_class: torch.nn.NLLLoss,
}

categorical_classification_recipe = {
    classes: True,
    use_weights: True,
    output_dim: 6,
    skip_logsoftmax: True,
    recipe_name: "Classifier",
    classification_type: "categorical",
    loss_fn_class: torch.nn.CrossEntropyLoss,
}

ordinal_classification_recipe = {
    classes: True,
    use_weights: True,
    output_dim: 5,
    skip_logsoftmax: True,
    recipe_name: "OrdinalClassifier",
    classification_type: "corn",
    loss_fn_class: tw.CORNLogitsLoss,
    weight_type: "cum_p",
}

regression_recipe = {
    classes: False,
    use_weights: False,
    output_dim: 1,
    recipe_name: "Regressor",
}

hidden_dim = "hidden_dim"
transformer_layers = "transformer_layers"
attention_heads = "attention_heads"


embedder = "embedder"
cnn_repeats = "cnn_repeats"
hidden_layers = "hidden_layers"
kernel = "kernel"
stride = "stride"
channel_dim = "channel_dim"
embed_shape = "embed_shape"
depth = "depth"
pad_depth = "pad_depth"

base_model_recipe = {
    hidden_layers: 2,
    cnn_repeats: 0,
    kernel: 5,
    stride: 1,
    depth: 4,
    pad_depth: 4,
}

transcoder_recipe = {
    model_name: "Transcoder",
    embedder: tm.TranscoderEmbedder,
    transformer_layers: 2,
    attention_heads: 4,
    channel_dim: 32,
    hidden_dim: 32,
    embed_shape: (64,64),
    kernel: 3,
}

tdnn_recipe = {
    model_name: "BasicTDNN",
    embedder: tm.BasicTDNNEmbedder,
    channel_dim: 32,
    hidden_dim: 16,
    kernel: 3,
}

at_tdnn_recipe_small = {
    model_name: "ATTDNN",
    embedder: tm.ATTDNNEmbedder,
    recipe_name: "Small",
    hidden_dim: 32,
    transformer_layers: 2,
    attention_heads: 6,
    embed_shape: (64,64),
}

at_tdnn_recipe_large = {
    model_name: "ATTDNN",
    embedder: tm.ATTDNNEmbedder,
    recipe_name: "Large",
    hidden_dim: 64,
    transformer_layers: 2,
    attention_heads: 6,
    embed_shape: (128,128),
}

transformer_tdnn_recipe = {
    model_name: "TransformerTDNN",
    embedder: tm.TransformerTDNNEmbedder,
    hidden_dim: 64,
    transformer_layers: 2,
    attention_heads: 6,
    dims: 1,
}

def update_recipe(recipe_1, recipe_2):
    new_recipe_name = (recipe_1.get(recipe_name) or "") + (recipe_2.get(recipe_name) or "")
    new_recipe = {**recipe_1, **recipe_2}
    new_recipe[recipe_name] = new_recipe_name
    return new_recipe

def add_all_recipes(*recipes):
    collated_recipe = {}
    for recipe in recipes:
        collated_recipe = update_recipe(collated_recipe, recipe)
    return collated_recipe