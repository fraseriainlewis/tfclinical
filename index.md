![](reference/figures/logo.png) tfclinical

This package is a collection of multilingual (R and Python) vignettes
showing how to use [Tensorflow Probability
(TPF)](https://www.tensorflow.org/probability) in
[RStudio](https://posit.co/downloads/). The focus is modelling building
and trial simulation for clinical research and drug development
activities.

## Why Tensorflow Probability?

[Tensorflow Probability](https://www.tensorflow.org/probability) (TFP)
is an alternative to [rstan](https://mc-stan.org/rstan/). Having two
major open source numerical libraries offering similar functionality
reduces risk. rstan is the more mature library for Bayesian modelling,
while TFP being part of Google’s
[Tensorflow](https://www.tensorflow.org/) framework offers computational
flexibility (CPU/GPU/TPU).

## Features

The vignettes range from demonstrating basic step-by-step mechanics of
running MCMC in RStudio using TFP, for example how to use **adaptive
step sizes** and **multiple chains**, through to complete examples of
simulating Bayesian basket trials using TFP and comparing the results
against rstan. Most vignettes include a comparison with
[rstan](https://mc-stan.org/rstan/).

## Installation overview

Due to the computationally intensive nature of the code the vignettes
show precomputed outputs, but the vignettes contain all the necessary
code to repeat the computations. To run the computations the
[tfclinical](https://github.com/fraseriainlewis/tfclinical.git) library
needs installed and it has one main dependency
[tfprobability](https://rstudio.github.io/tfprobability/index.html). The
tfprobability library is not directly used but installing this (see
[installation
instructions](https://rstudio.github.io/tfprobability/index.html)
ensures the necessary Python libraries are available. There is one
exception, in that we also use the Python pandas library and so this
should be included in the tfprobabilty install script (see below) in the
\`\`extras’’ option.

### Installation details for Linux

Python needs to be installed, and pyenv can also be a very useful tool
if dealing with multiple Python version and to avoid impacting the
system Python installation.

``` r
##########################################################
# in RStudio Terminal (not console) or Bash
sudo apt-get update
sudo apt-get install python3-venv python3-pip python3-dev
##########################################################
# in RStudio console
install.packages("tensorflow")
library(tensorflow)
install_tensorflow(extra_packages = c("tf_keras", "tensorflow", "tensorflow-probability","pandas"))
#restart session
install.packages("tfprobability")
library(tensorflow)
library(tfprobability)
d <- tfd_binomial(total_count = 7, probs = 0.3) # if this works then tensorflow us correctly installed
##########################################################
```

### Installation details for Linux

On Windows the key part is to have a suitable python installation and
also that RStudio can locate this. Once this is in place then the same
installation applied inside RStudio as for the above Linux case.

### Installation of separate Python venv

For the Graphical Neural Network (GNN) vignette this requires a separate
Python venv to be setup because some tensorflow projects, such as
TF-GNN, have very strict compatibility requirements which likely need
install versions hardcoded. This has only been tested on Linux and
MacOS.

``` bash
# in bash
python3 -m venv gnn
source gnn/bin/activate
# now install via pip the specific library versions needed
pip install tensorflow==2.16.2 tf_keras==2.16.0 tensorflow-gnn
```

The second part is to instruct reticulate to use this Python venv rather
than one of the existing virtualenv environments, e.g. that created when
tfprobability is installed.

``` r
# in RStudio console
library(reticulate)
use_virtualenv("/Users/work/gnn", required = TRUE) # tell R to use the python interpreter and libraries in here
```
