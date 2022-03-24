# Classification

**Classification:**
The idea is to measure the distance between the points we want to predict and the points that we already know.
-> Measure using the geometric distance

```r
distance <- sqrt((xa - xb)^2 + (ya - yb)^2)

or

distance <- (point_a - point_b)^2 %>%  #point_a / point_b are vectors
  sum() %>%
  sqrt()
  
or

dist_cancer_two_rows <- cancer  %>% 
    slice(1,2)  %>% 
    select(Symmetry, Radius, Concavity)  %>% 
    dist() # use dist () fn to figure out the distance

```

**Recipe:**
As seen below, recipe is used when describing how we want to do the prediction. We first have to specify what is the variable we want to predict and the predictors we are using, and then select the data (usually the training set).
```r
fruit_recipe <- recipe(fruit_name ~ mass + color_score, data = fruit_train) %>%
    step_scale(all_predictors()) %>%
    step_center(all_predictors())
```
If we want to take a look at the scaled data, the following steps are required:
```r
fruit_scaled <- fruit_recipe %>%
    prep () %>%
    bake(fruit_data)
```
Note that we have to specify the data frame that we want to bake (= the data set on which we can carry out our modifications in the recipe)

**Model specification:**
almost always the same for K-nn analysis:
```r
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = x) %>% #neighbors = tune() for vfold
       set_engine("kknn") %>%
       set_mode("classification") # set mode to "regression" if doing K-nn regression
```

**Training the classifier:**
This is done through the function ``` workflow() ```
```r
fruit_fit <- workflow() %>%
       add_recipe(fruit_recipe) %>%
       add_model(knn_spec) %>%
       fit(data = fruit_train)
```
Note that the data set in ```fit()``` is the training set. The testing set comes into play in predict().

**Tuning:**
In general it is very similar to a K-nn classification where we use known # of neighbors. A chunk of code is included with the difference commented.
```r
number_vfold <- vfold_cv(training_set, v = 5, strata = y) # perform  x-fold cross-validation, x = v

knn_tune <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>%
       set_engine("kknn") %>%
       set_mode("classification"
       
knn_results <- workflow() %>%
       add_recipe(number_recipe) %>%
       add_model(knn_tune) %>%
       tune_grid(resamples = number_vfold, grid = 10) %>% # additional step compared to normal K-nn
       collect_metrics() # collect the results for the tuning process, and determine which k value we shall use
accuracies <- knn_results %>% 
        filter(.metric == "accuracy") %>%
        filter(mean == max(mean)) # to figure out which one is the most ideal model
```
Training set and testing set: we only use the testing set when we want to do the actually prediction (with the function ```predict()```). Before this step, we only use the training set.

# Individual fn's
```
prep()
bake(data_frame)
```
These specific fucntions are used when we want to obtain the data frame in the recepie. When using ```workflow()```, neither is necessary
```
predict(data_frame, fit_model)
```
predict(workflow result, vector)
```
bind_cols(data_frame)
```
combines a newly outputted column with the data set in the ()
```
collect_metrics()
metrics()
```
```collect_metrics()``` is used for collecting results for the tuning process (= the output is for all the k values that we used in the tuning process), while ```metrics()``` is used to see how accurate our prediction is (single prediction), and also the RMSE values for regression
```
pull()
```
extract values in certain colomn as numerics
```
as_numeric()
```
```
conf_mat()
```
Similar to metrics, but gives a confusion matrix instead.
```
filter(mean == max(mean))
```


# Chunks of code:
**Splitting the Dataset**
```r
marathon_split <- initial_split(marathon, prop = 0.75, strata = time_hrs)
marathon_training <- training(marathon_split)
marathon_testing <- testing(marathon_split)
```
**Classification with known # of neighbors：**
```r
new_seed <- tibble(area = 12.1,
                        perimeter = 14.2,
                        compactness = 0.9,
                        length = 4.9,
                        width = 2.8,
                        asymmetry_coefficient = 3.0, 
                        groove_length = 5.1)

seed_data <- read_table2("data/seeds_dataset.txt")

colnames(seed_data) <- c("area", "perimeter", "compactness", "length",
                        "width", "asymmetry_coefficient", "groove_length", "Category") # Set up the object we want to classify
                        
seed_data_1 <- mutate(seed_data, Category = factor(Category))

knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 5) %>%
    set_engine("kknn") %>%
    set_mode("classification")

seed_recipe <- recipe(Category ~ ., data = seed_data_1) %>%
    step_scale(all_predictors()) %>%
    step_center(all_predictors())

seed_fit <- workflow() %>%
                add_recipe(seed_recipe)%>%
                add_model(knn_spec) %>%
                fit(data = seed_data_1)

seed_predict <- predict(seed_fit, new_seed) # the output of workflow can be directly used in the function ```predict()```
```

**Classification with tuning:**
```r
number_recipe <- recipe(y ~., data = training_set)

knn_tune <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>%
       set_engine("kknn") %>%
       set_mode("classification")

number_vfold <- vfold_cv(training_set, v = 5, strata = y)

knn_results <- workflow() %>%
       add_recipe(number_recipe) %>%
       add_model(knn_tune) %>%
       tune_grid(resamples = number_vfold, grid = 10) %>%
       collect_metrics()

accuracies <- knn_results %>% 
        filter(.metric == "accuracy")
```
Graphing the cross evaluation plot to figure out the best k value
```r
cross_val_plot <- ggplot(accuracies, aes(x = neighbors, y = mean))+
       geom_point() +
       geom_line() +
       labs(x = "Neighbors", y = "Accuracy Estimate") + 
       scale_x_continuous(breaks = seq(0, 14, by = 1)) # adjusting the x-axis
```
```r
mnist_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = 3) %>%
    set_engine("kknn") %>%
    set_mode("classification")

mnist_fit <- workflow() %>%
        add_recipe(number_recipe)%>%
        add_model(mnist_spec) %>%
        fit(data = training_set)

mnist_predictions <- predict (mnist_fit, testing_set) %>%
    bind_cols(testing_set)

mnist_metrics <- mnist_predictions %>%
    metrics(truth = y, estimate = .pred_class)

mnist_conf_mat <- mnist_predictions %>% 
    conf_mat(truth = y, estimate = .pred_class)
```

**K-nn regression:**
```r
credit_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) %>% 
       set_engine("kknn") %>%
       set_mode("regression") # we are doing a regression here instead of a classification!!

credit_recipe <- recipe(Balance ~ ., data = credit_training) %>%
       step_scale(all_predictors()) %>%
       step_center(all_predictors())

credit_vfold <- vfold_cv(credit_training, v = 5, strata = Balance)

gridvals <- tibble(neighbors = seq(from = 1, to = 20)) # use k from a 1 to 20

credit_results<- workflow() %>%
       add_recipe(credit_recipe) %>%
       add_model(credit_spec) %>%
       tune_grid(resamples = credit_vfold, grid = gridvals) %>%
       collect_metrics() 
```
Up to this step, it is very similar compared to how we do a classification! The main difference is that we are using the "regression" mode of in our specification instead of classification.
```r
credit_min <- credit_results %>%
    filter(.metric == "rmse") %>%
    filter(mean == min(mean)) 

k_min <- credit_min %>%
    pull(neighbors) # We managed to choose the ideal # of neighbors that we are going to use for the regression as k_min

credit_best_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = k_min) %>%
          set_engine("kknn") %>%
          set_mode("regression") # build the specification, notice that we use "regression" instead of "classification"

credit_best_fit <- workflow() %>%
          add_recipe(credit_recipe) %>%
          add_model(credit_best_spec) %>%
          fit(data = credit_training) # fit our data onto the training set in order to figure out the slope and intercept.

credit_summary <- credit_best_fit %>%
           predict(credit_testing) %>%
           bind_cols(credit_testing) %>%
           metrics(truth = Balance, estimate = .pred) # use our model obtained from the training set onto the testing set, thereby obtaining the RMSPE

knn_rmspe <- credit_summary %>%
    filter(.metric == "rmse") %>%
    pull(.estimate) # to get the RMSE for the prediction
```
