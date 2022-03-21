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



# Individual fn's
```
pull()
```
```
as_numeric()
```
