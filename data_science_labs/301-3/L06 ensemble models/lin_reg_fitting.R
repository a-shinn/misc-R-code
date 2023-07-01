# Linear model ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# load required objects ----
load("model_info/wildfires_recipe.rda")
load("data/wildfires_folds.rda")

# Define model ----
lin_reg_model <- linear_reg(
    mode = "regression",
  ) %>%
  set_engine("lm")

# workflow ----
lin_reg_workflow <- workflow() %>%
  add_model(lin_reg_model) %>%
  add_recipe(wildfires_recipe)

# Tuning/fitting ----
lin_reg_res <- lin_reg_workflow %>%
  fit_resamples(
    resamples = wildfires_folds,
    control = stacks::control_stack_resamples()
    )

# Write out results & workflow
save(lin_reg_res, file = "model_info/lin_reg_res.rda")


