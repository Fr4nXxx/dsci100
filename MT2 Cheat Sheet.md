# DSCI 100

**Classification:**

The idea is to measure the distance between the points we want to predict and the points that we already know.
-> Measure using the geometric distance

```
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
```
fruit_recipe <- recipe(fruit_name ~ mass + color_score, data = fruit_train) %>%
    step_scale(all_predictors()) %>%
    step_center(all_predictors())
```

**Model specification:**
almost always the same for K-nn analysis:
```
knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = x ) %>% #neighbors = tune() for vfold
       set_engine("kknn") %>%
       set_mode("classification")
```




# Individual fn's
```
pull()
```
```
as_numeric()
```
