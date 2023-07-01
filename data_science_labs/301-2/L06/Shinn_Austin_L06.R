# Load packages here!
library(tidyverse)
library(tidymodels)
library(janitor)
library(skimr)
library(patchwork)

# Set seed here!
set.seed(77)

# exercise 1
kc_house <- read_csv("data/kc_house_data.csv") %>%
  clean_names()

# exercise 2
p1 <- ggplot(kc_house, aes(price)) +
  geom_density()

p2 <- ggplot(kc_house, aes(price)) +
  geom_boxplot() 

p2 / p1

kc_house <- kc_house %>% mutate(price = log10(price))

p3 <- ggplot(kc_house, aes(price)) +
  geom_density()

p4 <- ggplot(kc_house, aes(price)) +
  geom_boxplot() 

p4 / p3

skim_without_charts(kc_house)


# exercise 3
kc_splits <- initial_split(kc_house, prop = 0.8, strata = price)
dim(kc_splits)


kc_train <- training(kc_splits)
dim(kc_train)

kc_test <- testing(kc_splits)
dim(kc_test)

# exercise 4
kc_folds <- vfold_cv(kc_train, v = 5, repeats = 3)

# exercise 6
kc_recipe <- recipe(price ~ . , data = kc_train) %>%
  step_rm(id, date, zipcode) %>%
  step_log(sqft_living, sqft_lot, sqft_above, sqft_living15, sqft_lot15, base = 10) %>%
  step_normalize(all_predictors())

# exercise 7
# model 1
linearm <- linear_reg() %>%
  set_mode("regression") %>%
  set_engine("lm")

lin_workflow <- workflow() %>%
  add_recipe(kc_recipe) %>%
  add_model(linearm)

# model 2
ridgem <- linear_reg(penalty = 0.1, mixture = 0) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

ridge_workflow <- workflow() %>%
  add_recipe(kc_recipe) %>%
  add_model(ridgem)

# model 3
lassom <- linear_reg(penalty = 0.1, mixture = 1) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

lasso_workflow <- workflow() %>%
  add_recipe(kc_recipe) %>%
  add_model(lassom)

# model 4
rfm <- rand_forest(trees = 600, min_n = 10) %>%
  set_mode("regression") %>%
  set_engine("ranger")

rf_workflow <- workflow() %>%
  add_recipe(kc_recipe) %>%
  add_model(rfm)

kc_control <- control_resamples(save_pred = TRUE)

# exercise 8
# model 1
lin_fit_folds <- fit_resamples(
  lin_workflow, resamples = kc_folds, control = kc_control)

# model 2
ridge_fit_folds <- fit_resamples(
  ridge_workflow, resamples = kc_folds, control = kc_control)

# model 3
lasso_fit_folds <- fit_resamples(
  lasso_workflow, resamples = kc_folds, control = kc_control)

rf_fit_folds <- fit_resamples(
  rf_workflow, resamples = kc_folds, control = kc_control)

# exercise 9
# model 1
collect_metrics(lin_fit_folds)

# model 2
collect_metrics(ridge_fit_folds)

# model 3
collect_metrics(lasso_fit_folds)

# model 4
collect_metrics(rf_fit_folds)

# exercise 10
rf_training_fit <- rf_workflow %>%
  fit(kc_train)

# exercise 11
kc_metrics <- metric_set(rmse, rsq)

rf_training_fit %>% 
  predict(new_data = kc_test) %>%
  bind_cols(kc_test %>% select(price)) %>%
  kc_metrics(truth = price, estimate = .pred)