% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/arithmetic.R
\name{adaboost_rt}
\alias{adaboost_rt}
\title{adaboost regression tree}
\usage{
adaboost_rt(data, outcome_var, M)
}
\arguments{
\item{data}{we split the original data randomly into 2 datasets, one is train data}

\item{outcome_var}{response variable or target variable of the dataset}

\item{M}{the number of decision trees included in the ensemble}
}
\value{
ensemble trees
}
\description{
adaboost regression tree
}
\examples{
library(rpart)
library(tidyr)
library(tidyverse)
library(gbm)
library(caret)
library(palmerpenguins)
data("penguins")
penguins <- penguins \%>\% drop_na()
set.seed(123)
indices <- sample(1:nrow(penguins), 0.7 * nrow(penguins))
train_data <- penguins[indices, ]
test_data <- penguins[-indices, ]
outcome_variable <- "body_mass_g"
models <- adaboost_rt(train_data, outcome_variable, 100)
}
