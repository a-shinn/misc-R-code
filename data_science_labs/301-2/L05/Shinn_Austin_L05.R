# Load packages here!
library(tidyverse)
library(tidymodels)
library(ranger)
# Set seed here!
set.seed(77)

# exercise 1
titanic_data <- read.csv('data/titanic.csv')

titanic_data$survived <- factor(titanic_data$survived, levels = c("Yes", "No"))

titanic_data$pclass <- factor(titanic_data$pclass, levels = c(1,2,3), labels = c("1st", "2nd", "3rd"))

# exercise 2
# total counts of survivors and fatalities
titanic_data %>%
  group_by(survived) %>%
  summarise(count = n()) 

# proportions of survivors among groups
titanic_data %>%
  mutate(survived = ifelse(survived == "Yes", 1, 0)) %>%
  group_by(survived) %>%
  summarise(count = n()) %>%
  summarise(prop = count/sum(count))

# exercise 3
# split data into 80/20
titanic_split <- initial_split(data = titanic_data, prop = 0.80, strata = survived)

# training and testing
titanic_training <- training(titanic_split)
titanic_testing <- testing(titanic_split)

# skim data
skimr::skim(titanic_training)

# exercise 4
log_reg_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_training) %>%
  step_impute_linear(age) %>%
  step_dummy(sex, pclass) %>%
  step_interact(~ starts_with("sex"):fare + age:fare)

tree_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_training) %>%
  step_impute_linear(age) %>%
  step_dummy(sex, pclass, one_hot = TRUE) 

# exercise 5
# set engine to logistic
model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# create workflow
log_workflow <- workflow() %>%
  add_model(model) %>%
  add_recipe(log_reg_recipe) 

# fit the logistic regression
log_fit <- fit(log_workflow, titanic_testing)

# exercise 6
tree_model <- rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("classification")

# create workflow
tree_workflow <- workflow() %>%
  add_model(tree_model) %>%
  add_recipe(tree_recipe) 

# fit random forest
tree_fit <- fit(tree_workflow, titanic_testing)

# exercise 7
tree_model2 <- rand_forest(mtry = 4, trees = 1000, min_n = 3) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# create workflow
tree_workflow2 <- workflow() %>%
  add_model(tree_model2) %>%
  add_recipe(tree_recipe) 

# fit random forest
tree_fit2 <- fit(tree_workflow2, titanic_testing)

# exercise 8
titanic_metrics <- metric_set(accuracy)

# fit predictions for logistic, default random forest, and custom random forest
log_pred <- predict(log_fit, new_data = titanic_testing %>% select(sex, age, pclass, sib_sp, parch, fare)) %>%
  bind_cols(titanic_testing %>% select(survived))

tree_pred <- predict(tree_fit, new_data = titanic_testing %>% select(sex, age, pclass, sib_sp, parch, fare)) %>%
  bind_cols(titanic_testing %>% select(survived))

tree_pred2 <- predict(tree_fit2, new_data = titanic_testing %>% select(sex, age, pclass, sib_sp, parch, fare)) %>%
  bind_cols(titanic_testing %>% select(survived))

# calculate and display metrics
titanic_metrics(log_pred, truth = survived, estimate = .pred_class)
titanic_metrics(tree_pred, truth = survived, estimate = .pred_class)
titanic_metrics(tree_pred2, truth = survived, estimate = .pred_class)

# exercise 9
conf_mat(tree_pred2, truth = survived, estimate = .pred_class)

# exercise 10
tree_pred2 <- predict(tree_fit2, new_data = titanic_testing %>% select(sex, age, pclass, sib_sp, parch, fare), type = "prob") %>%
  bind_cols(titanic_testing %>% select(survived))

tree_pred2

# exercise 11
two_class_curve <- roc_curve(tree_pred2, survived, .pred_Yes)

# calculate area under curve
roc_auc(tree_pred2, survived, .pred_Yes)

# plots the curve
autoplot(two_class_curve)