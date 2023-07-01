# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load("model_info/knn_res.rda")
load("model_info/svm_res.rda")
load("model_info/lin_reg_res.rda")

# Load split data object & get testing data
load("data/wildfires_split.rda")

wildfires_test <- wildfires_split %>% testing()

# Create data stack ----
wildfires_data_stack <- stacks() %>%
  add_candidates(knn_res)  %>%
  add_candidates(svm_res)  %>%
  add_candidates(lin_reg_res)


# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(777)

wildfires_blend <- wildfires_data_stack %>%
  blend_predictions(penalty = blend_penalty)

# Save blended model stack for reproducibility & easy reference (Rmd report)
save(wildfires_blend, file = "model_info/wildfires_blend.rda")

# Explore the blended model stack
autoplot(wildfires_blend, type = "weights") +
  theme_minimal( )

autoplot(wildfires_blend, type = "members") +
  theme_minimal( )

# fit to ensemble to entire training set ----
wildfires_final <- wildfires_blend %>%
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (Rmd report)
save(wildfires_final, file = "model_info/wildfires_final.rda")

# Explore and assess trained ensemble model
collect_parameters(wildfires_final, "svm_res")

wildfires_final

autoplot(wildfires_final, type = "weights") +
  theme_minimal( )

autoplot(wildfires_final, type = "members") +
  theme_minimal()
