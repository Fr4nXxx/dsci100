**Linear regression**

Splitting the data set
```r
credit_split <- initial_split(credit, prop = 0.6, strata = Balance)
credit_training <- training(credit_split)
credit_testing <- testing(credit_split)
```
```r
lm_spec <- linear_reg() %>%
    set_engine("lm") %>%
    set_mode("regression")

credit_recipe <- recipe(Balance ~ ., data = credit_training)
```
Spec and recipe
```r
credit_fit <- workflow() %>%
       add_recipe(credit_recipe) %>%
       add_model(lm_spec) %>%
       fit(credit_training)
```
Fitting - print to get the formula
```r
lm_rmse <- credit_fit %>%
         predict(credit_training) %>%
         bind_cols(credit_training) %>%
         metrics(truth = Balance, estimate = .pred) %>%
         filter(.metric == "rmse") %>%
         select(.estimate) %>%
         mutate(.estimate = as.numeric(.estimate)) %>%
         pull()
lm_rmse
```
Getting the RMSE value - change the dataset into credit_testing for RMPSE value
