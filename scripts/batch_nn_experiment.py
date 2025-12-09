import data_logger as dl
import model_runner as mr
import torch
import torch_layers as tl
import keywords_recipes as kwr

cuda_supressed = False
device = None
if torch.cuda.is_available() and not cuda_supressed:
    device="cuda"
    dl.log("Using CUDA enabled GPU.")


goal_recipes = [kwr.log_categorical_classification_recipe, kwr.ordinal_classification_recipe, kwr.regression_recipe]
model_recipes = [kwr.tdnn_recipe, kwr.transformer_tdnn_recipe]
override_recipe = {
    kwr.device: device,
}

goal_recipe = goal_recipes[1]
for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    for wd in [0,1e-4,1e-5,1e-6]:
        for model_recipe in model_recipes:
            recipe_name = f"_lr{lr}_wd{wd}"
            tuning_recipe = {
                kwr.batch_size: 128,
                kwr.dropout_rate: 0.4,
                kwr.learning_rate: lr,
                kwr.weight_decay: wd,
                kwr.epochs: 20,
                kwr.context: 16,
                kwr.iterations: 1,
                kwr.skip_norm: True,
                kwr.skip_posenc: False,
                kwr.skip_se: False,
                kwr.skip_preembed: True,
                kwr.pooling_kernel: (1,4),
                kwr.recipe_name: recipe_name,
            }
            
            recipe = kwr.add_all_recipes(
                kwr.gen_recipe, 
                kwr.data_recipe, 
                kwr.learning_recipe, 
                kwr.base_recipe, 
                kwr.hyperparameter_recipe, 
                goal_recipe, 
                kwr.base_model_recipe, 
                model_recipe, 
                tuning_recipe, 
                override_recipe
            )

            tasks_info = ", ".join(recipe.get(kwr.tasks))
            data_info = f"from \'{recipe.get(kwr.label_file)}\'"

            dl.log(f"Started training a neural network on {tasks_info} with data {data_info}.")
            try:
                mr.experiment_model(recipe)
                dl.log("Finished training neural network.")
            except:
                dl.log_stack("Failed to train neural network.")