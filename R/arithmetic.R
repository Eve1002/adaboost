#' adaboost regression tree
#'
#' @param data we split the original data randomly into 2 datasets, one is train data
#' @param outcome_var response variable or target variable of the dataset
#' @param M the number of decision trees included in the ensemble
#'
#' @return ensemble trees
#' @export
#'
#' @examples
#' library(rpart)
#' library(tidyr)
#' library(tidyverse)
#' library(gbm)
#' library(caret)
#' library(palmerpenguins)
#' data("penguins")
#' penguins <- penguins %>% drop_na()
#' set.seed(123)
#' indices <- sample(1:nrow(penguins), 0.7 * nrow(penguins))
#' train_data <- penguins[indices, ]
#' test_data <- penguins[-indices, ]
#' outcome_variable <- "body_mass_g"
#' models <- adaboost_rt(train_data, outcome_variable, 100)
adaboost_rt <- function(data, outcome_var, M) {

  y <- data[[outcome_var]]
  n <- length(y)
  weights <- rep(1/n, n)
  models <- list()

  for (m in 1:M) {
    model <- rpart(formula = as.formula(paste(outcome_variable, "~ .")), data = data, weights = weights)
    predictions <- predict(model, data)

    # Calculate Loss function L
    diff <- abs(y - predictions)
    L <- diff/max(diff)
    epsilon <- sum(weights * L)

    # Calculate beta to update the model weight
    beta <- epsilon/(1 - epsilon)

    # Update the weights
    weights <- weights * ( beta ^ (1-L) )

    # Normalize the weights
    weights <- weights / sum(weights)

    models[[m]] <- list(model)
  }

  return(models)
}



#' use cross-validation to tune hyperparameter M
#'
#' @param data we split the original data randomly into 2 datasets, one is train data,
#' @param outcome_var response variable or target variable of the dataset
#' @param M the number of decision trees included in the ensemble
#' @param folds how many folds we use
#'
#' @return average mean squared error across folds
#' @export
#'
#' @examples
#' library(rpart)
#' library(tidyr)
#' library(tidyverse)
#' library(gbm)
#' library(caret)
#' library(palmerpenguins)
#' data("penguins")
#' penguins <- penguins %>% drop_na()
#' set.seed(123)
#' indices <- sample(1:nrow(penguins), 0.7 * nrow(penguins))
#' train_data <- penguins[indices, ]
#' test_data <- penguins[-indices, ]
#' outcome_variable <- "body_mass_g"
#' M_values <- c(5, 10, 15, 20, 30, 50, 70)
#' cv_results <- c()
#' for(m in 1:length(M_values)){cv_results[m] <- adaboost_cv(train_data, outcome_variable, M=M_values[m], folds = 5)}
#' best_M <- M_values[which.min(cv_results)]
adaboost_cv <- function(data, outcome_var, M, folds = 5) {
  set.seed(123)
  y <- data[[outcome_var]]
  n <- nrow(data)
  size <- floor(n / folds)
  indices <- sample(1:n)
  mse1 <- c()
  mse <- c()

  for (i in 1:folds) {
    # Split data into training and validation sets
    validation_indices <- indices[((i-1) * size + 1):(i * size)]
    train_indices <- setdiff(indices, validation_indices)

    train_data <- data[train_indices, ]
    validation_data <- data[validation_indices, ]

    # Train AdaBoost model on training data
    models <- adaboost_rt(train_data, outcome_var, M)

    # Make predictions on validation data
    predictions <- predict(models, validation_data)

    for(m in 1:length(models)){
      mse1[m] <- mean((validation_data[[outcome_var]] - predictions[[m]][[1]]) ^ 2)
    }

    mse[i] <- mse1[length(models)]
  }

  # Return the average mean squared error across folds
  return(mean(mse))
}
