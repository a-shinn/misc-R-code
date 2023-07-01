# Knn tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(2021)

# load required objects ----
load("model_info/wildfires_recipe.rda")
load("data/wildfires_folds.rda")

# Define model ----
knn_model <- nearest_neighbor(
  mode = "regression",
  neighbors = tune()
) %>%
  set_engine("kknn")

# # check tuning parameters
parameters(knn_model)

# set-up tuning grid ----
knn_params <- parameters(knn_model) %>%
  update(neighbors = neighbors(range = c(1,40)))

# define grid
knn_grid <- grid_regular(knn_params, levels = 15)

# workflow ----
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
knn_res <- knn_workflow %>%
  tune_grid(
    resamples = wildfires_folds,
    grid = knn_grid,
    control = stacks::control_stack_grid()
  )

# Write out results & workflow
save(knn_res, file = "model_info/knn_res.rda")


