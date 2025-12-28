# my-vignette-test

------------------------------------------------------------------------

## Bayesian

ddd \### dddd

## Dataset

``` r
### Prepare model inputs
set.seed(9999)
# Set up data
rr_k_ctrl <- c(0.60)       # control response rate for each basket
rr_k_trt <- c(0.58)        # treatment response rate for each basket

K<-length(rr_k_ctrl)             # number of baskets

N_k_ctrl <- rep(250, K)     # number of control participants per basket
N_k_trt <- rep(250, K)      # number of treatment participants per basket
N_k <- N_k_ctrl + N_k_trt         # number of participants per basket (both arms combined)
N <- sum(N_k)                     # total sample size
k_vec <- rep(1:K, N_k)            # N x 1 vector of basket indicators (1 to K)

z_vec<-NULL;
y<-NULL;
for(i in 1:K){ # for each basket repeat 0-control 1-trt inside this according to the specifc Ns
  z_vec<-c(z_vec,rep(0:1,c(N_k_ctrl[i],N_k_trt[i]))) # treatment/control indicator
  y<-c(y,
       c(rbinom(N_k_ctrl[i],1,rr_k_ctrl[i]), # bernoulli for control
         rbinom(N_k_trt[i],1,rr_k_trt[i]))) #           for trt
}

thedata<-data.frame(y,basketID=k_vec,Treatment=z_vec)
knitr::kable(head(thedata)) 
```

|   y | basketID | Treatment |
|----:|---------:|----------:|
|   0 |        1 |         0 |
|   0 |        1 |         0 |
|   0 |        1 |         0 |
|   1 |        1 |         0 |
|   0 |        1 |         0 |
|   0 |        1 |         0 |

``` r
py$thedata<-r_to_py(thedata) # to ensure correct form to python
#> Downloading uv...Done!
```

## 2. Fit Bayesian Model in Tensorflow - One basket, control/test, binary endpoint

This runs python via reticulate with dependencies previously installed
as part of the package. Note that the dataset created above has already
been passed to python via reticulate r_to_py() function, and it is
accessible as a pandas data.frame inside python chunks.
