# Load package(s)
library(tidymodels)
library(tidyverse)

# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(777)

# Load data ----
wildfires_dat <- read_csv("data/wildfires.csv") %>%
  janitor::clean_names() %>%
  mutate(
    winddir = factor(winddir, levels = c("N", "NE", "E", "SE", "S", "SW", "W", "NW")),
    traffic = factor(traffic, levels = c("lo", "med", "hi")),
    wlf = factor(wlf, levels = c(1, 0), labels = c("yes", "no"))
  ) %>%
  select(-wlf)

# Data checks ----
# Outcome/target variable

ggplot(wildfires_dat, aes(x = burned)) +
  geom_histogram(bins = 50)

# Some outliers, but not too much and outcome is not very skewed
# no missing values (in the whole dataset)
skimr::skim(wildfires_dat$burned)


# check missingness & look for extreme issues
naniar::miss_var_summary(wildfires_dat)
skimr::skim_without_charts(wildfires_dat)

# Initial split & folding ----
wildfires_split <- wildfires_dat %>%
  initial_split(prop = 0.75, strata = burned)

wildfires_train <- training(wildfires_split)

# Folds
wildfires_folds <- vfold_cv(wildfires_train, v = 5, repeats = 3, strata = burned)

## Build general recipe (featuring eng.) ----
wildfires_recipe <- recipe(burned ~ . , data = wildfires_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# # Check how recipe works
 wildfires_recipe %>%
   prep(wildfires_train) %>%
   bake(new_data = NULL)

# objects required for tuning
# data objects
save(wildfires_folds, file = "data/wildfires_folds.rda")
save(wildfires_split, file = "data/wildfires_split.rda")

# model info object
save(wildfires_recipe, wildfires_split, file = "model_info/wildfires_recipe.rda")

