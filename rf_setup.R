# load package(s)
library(tidymodels)
library(tidyverse)

# set seed
set.seed(57)

## load training set
loan_train <- read_csv("data/train.csv") %>% 
  mutate(hi_int_prncp_pd = factor(hi_int_prncp_pd, levels = c("0", "1"))) %>% 
  mutate(across(where(is.character), as.factor))

## load testing set
loan_test <- read_csv("data/test.csv") %>% 
  mutate(across(where(is.character), as.factor))

## inspect data
skimr::skim_without_charts(loan_train)

loan_train %>% 
  mutate(hi_int_prncp_pd = as.numeric(hi_int_prncp_pd)) %>% 
  select(hi_int_prncp_pd, acc_now_delinq, acc_open_past_24mths, annual_inc, avg_cur_bal, bc_util, delinq_2yrs, delinq_amnt, dti, int_rate, loan_amnt, mort_acc, num_sats, num_tl_120dpd_2m, num_tl_30dpd, num_tl_90g_dpd_24m, out_prncp_inv, pub_rec, pub_rec_bankruptcies, tot_coll_amt, tot_cur_bal, total_rec_late_fee) %>% 
  cor(use = "complete.obs")

## create resamples
set.seed(57)
loan_fold <- vfold_cv(loan_train, v = 5, repeats = 3, strata = hi_int_prncp_pd)

## create recipe
loan_recipe <- recipe(hi_int_prncp_pd ~ int_rate + loan_amnt + out_prncp_inv + application_type + emp_length + grade + home_ownership + initial_list_status + sub_grade + term + verification_status, data = loan_train) %>% 
  step_other(emp_length, grade, sub_grade) %>% 
  step_nzv(all_nominal()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_nominal()) %>% 
  step_normalize(all_numeric())

## prep and bake recipe
loan_recipe %>% prep(loan_train) %>% bake(new_data = NULL)

## save objects required for tuning
save(loan_fold, loan_recipe, file = "data/loan_setup.rda")

## load tuning objects
load(file = "data/rf_tune.rda")

## train and fit model
workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "roc_auc"))

results <- fit(workflow_tuned, loan_train)

## predict using test set
predictions <- results %>%
  predict(new_data = loan_test) %>%
  bind_cols(loan_test %>% select(id)) %>%
  select(id, .pred_class) %>% 
  rename(Id = id) %>% 
  rename(Category = .pred_class)

write_csv(predictions, "data/predictions.csv")
