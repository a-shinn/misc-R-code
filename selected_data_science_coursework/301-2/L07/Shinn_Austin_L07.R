# Load packages here!
library(tidymodels)
library(ranger)
library(kknn)
library(xgboost)
library(skimr)

# Set seed here!
set.seed(777)

# exercise 1
carseats <- read.csv("data/carseats.csv") %>%
  mutate(across(where(is.character), factor))

# exercise 2
summary(carseats$sales)

ggplot(carseats, aes(sales)) +
  geom_histogram(bins = 20)

skim(carseats)

# exercise 3
initial <- initial_split(carseats, prop = .75, strata = sales)

carseats_train <- training(initial)
carseats_test <- testing(initial)

carseats_fold <- vfold_cv(carseats_train, v = 10, repeats = 5, strata = sales)

# exercise 4
recipe <- carseats_train %>%
  recipe(sales ~ .) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_normalize(all_predictors())

recipe %>%
  prep(training = carseats_train) %>%
  bake(new_data = NULL) %>%
  view()

save(carseats_fold, recipe, initial, file = "data/save.rda")

# exercise 5
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

bt_model <- boost_tree(mode = "regression",
                       mtry = tune(),
                       min_n = tune(),
                       learn_rate = tune()) %>%
  set_engine("xgboost")

knn_model <- nearest_neighbor(mode = "regression",
                              neighbors = tune()) %>%
  set_engine("kknn")

# exercise 6
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(c(2,10)))

rf_grid <- grid_regular(rf_params, levels = 5)

bt_params <- parameters(bt_model) %>%
  update(mtry = mtry(c(2,10)),
         learn_rate = learn_rate(c(-5, -.2)))

bt_grid <- grid_regular(bt_params, levels = 5)

knn_params <- parameters(knn_model)

knn_grid <- grid_regular(knn_params, levels = 5)

# exercise 7
rf_grid

# exercise 8
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(recipe)

bt_workflow <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(recipe)

knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe)

# exercise 9
rf_tune <- rf_workflow %>%
  tune_grid(resamples = carseats_fold,
            grid = rf_grid)

bt_tune <- bt_workflow %>%
  tune_grid(resamples = carseats_fold,
            grid = bt_grid)

knn_tune <- knn_workflow %>%
  tune_grid(resamples = carseats_fold,
            grid = knn_grid)

save(rf_tune, bt_tune, knn_tune, file = "data/carseats_tune_grids.rda")

# exercise 10
load("data/carseats_tune_grids.rda")

autoplot(rf_tune, metric = "rmse")

autoplot(bt_tune, metric = "rmse")

autoplot(knn_tune, metric = "rmse")

# exercise 11
show_best(rf_tune, metric = "rmse")

show_best(bt_tune, metric = "rmse")

show_best(knn_tune, metric = "rmse")

# exercise 12
bt_workflow_tuned <- bt_workflow %>% 
  finalize_workflow(select_best(bt_tune, metric = "rmse"))

bt_results <- fit(bt_workflow_tuned, carseats_train)

# exercise 13
carseat_metric <- metric_set(rmse)

predict(bt_results, new_data = carseats_test) %>% 
  bind_cols(carseats_test %>% select(sales)) %>% 
  carseat_metric(truth = sales, estimate = .pred)