

::: {style="font-size: 24px;"}
<!--<h1 style="display: flex; align-items: right; gap: 15px;"> -->
  <img src="man/figures/logo.png" style="height: 125px;">
  tfclinical
:::
  
<!--  tfclinical
</h1> -->

<!-- ::: {style="font-size: 17px;"} this hard codes font size, only way to do it -->
This package is a collection of heavily documented Rmarkdown vignettes demonstrating how to use [Tensorflow Probability (TPF)](https://www.tensorflow.org/probability) in [RStudio](https://posit.co/downloads/) for Bayesian inference. [rstan](https://mc-stan.org/rstan/) features in many of these for comparison. The application focus is model building and trial simulation for clinical research and drug development activities.


## Why Tensorflow Probability (TFP)?
  - Having alternative options (e.g., to [Stan](https://mc-stan.org/docs/reference-manual/mcmc.html)) when using open source reduces business risk 
  - [TFP](https://www.tensorflow.org/probability) is developed and maintained by Google as part of its [Tensorflow](https://www.tensorflow.org/) framework 
  - TFP provides a wide variety of building blocks for probabilistic modelling     

## Features
[TFP](https://www.tensorflow.org/probability) is less mature in terms of documentation compared [rstan](https://mc-stan.org/rstan/), in addition to having a Python API rather than R API. The vignettes included here are designed to help RStudio users get up to speed with TFP. Multi-lingual R markdown is used in each vignette to allow the strengths of each of R and Python to be used. There is an R wrapper library for TFP available (see [here](https://rstudio.github.io/tfprobability/index.html)).


## Running the Vignettes
Due to the computationally intensive nature of the code the vignettes show only pre-computed outputs. Links to the full live vignettes are included at the [tfclinical website](https://fraseriainlewis.github.io/tfclinical/). To run the live vignettes  [tfclinical](https://github.com/fraseriainlewis/tfclinical.git) needs installed and it has one main dependency [tfprobability](https://rstudio.github.io/tfprobability/index.html). The tfprobability library is not directly used but installing this (see [installation instructions](https://rstudio.github.io/tfprobability/index.html) ensures the necessary Python libraries are available. There is one exception, in that we also use the Python pandas library and so this should be included in the tfprobabilty install script (see below) in the ``extras'' option. 
<!-- ::: other pair from above-->

### Installation details for Linux
Python needs to be installed, and pyenv can also be a very useful tool if dealing with multiple Python version and to avoid impacting the system Python installation.  
```r
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
On Windows the key part is to have a suitable python installation and also that RStudio can locate this. Once this is in place then the same installation applied inside RStudio as for the above Linux case. 

### Installation of separate Python venv
For the Graphical Neural Network (GNN) vignette this requires a separate Python venv to be setup because some tensorflow projects, such as TF-GNN, have very strict compatibility requirements which likely need install versions hardcoded. This has only been tested on Linux and MacOS.
```bash
# in bash
python3 -m venv gnn
source gnn/bin/activate
# now install via pip the specific library versions needed
pip install tensorflow==2.16.2 tf_keras==2.16.0 tensorflow-gnn
```
The second part is to instruct reticulate to use this Python venv rather than one of the existing virtualenv environments, e.g. that created when tfprobability is installed.
```r
# in RStudio console
library(reticulate)
use_virtualenv("/Users/work/gnn", required = TRUE) # tell R to use the python interpreter and libraries in here
```


