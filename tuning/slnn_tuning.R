# Single layer neural network tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# load required objects ----
load("data/loan_setup.rda")

# define model ----
slnn_model <- mlp(
  mode = "classification", 
  hidden_units = tune(), 
  penalty = tune()
) %>% 
  set_engine("nnet")

# check tuning parameters
parameters(slnn_model)

# set-up tuning grid ----
slnn_params <- parameters(slnn_model)

# define tuning grid
slnn_grid <- grid_regular(slnn_params, levels = 5)

# workflow ----
slnn_workflow <- workflow() %>% 
  add_model(slnn_model) %>% 
  add_recipe(loan_recipe)

# tuning ----
tic("Single Layer Neural Network")
slnn_tune <- slnn_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = slnn_grid
  )
toc(log = TRUE)

# save runtime info
slnn_runtime <- tic.log(format = TRUE)

# write out results & workflow
save(slnn_tune, slnn_workflow, slnn_runtime, file = "data/slnn_tune.rda")
