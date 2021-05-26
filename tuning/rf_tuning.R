# Random forest tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load("data/loan_setup.rda")

# define model ----
rf_model <- rand_forest(
  mode = "classification", 
  mtry = tune(), 
  min_n = tune()
) %>% 
  set_engine("ranger")

# check tuning parameters
parameters(rf_model)

# set-up tuning grid ----
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(10, 20)))

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe)

# tuning ----
tic("Random Forest")
rf_tune <- rf_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = rf_grid
  )
toc(log = TRUE)

# save runtime info
rf_runtime <- tic.log(format = TRUE)

# write out results & workflow
save(rf_tune, rf_workflow, rf_runtime, file = "data/rf_tune.rda")
