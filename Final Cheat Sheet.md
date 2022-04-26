# Linear regression

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


# Clustering

Scaling the data, setting the number of centers (k-value)
```r
scaled_km_data<- km_data %>% 
    mutate(across(everything(), scale))
pokemon_clusters <- kmeans(scaled_km_data, centers = 4)
```
Clustering plot
```r
Clustering_plot <- augment(pokemon_clusters, scaled_km_data) %>%
    ggplot(aes(x = Speed, y = Defense)) +
    geom_point(aes(color = .cluster)) +
    labs(x = "Pokemon Speed Value", y = "Pokemon Defense Value", color = "Cluster")
```
Create "elbow plot" to figure out the best k value to use
```r
ks <- tibble(k = 1:10)
elbow_stats <- ks %>%
    rowwise() %>%
    mutate(poke_clusts = list(kmeans(scaled_km_data, nstart = 10, k))) %>%
    mutate(glanced = list(glance(poke_clusts))) %>%
    select(-poke_clusts) %>%
    unnest(glanced)
elbow_stats
```

# Inference
Sampling:
```r
samples_100 <- rep_sample_n(can_seniors, size = 100, reps = 1500)
```
Bootstrapping:
```r
boot1 <- one_sample %>% 
    rep_sample_n(size = 40, replace = TRUE, reps = 1)
```
