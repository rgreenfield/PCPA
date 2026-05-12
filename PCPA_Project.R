# PCPA Project Raymond Greenfield 3/21/2026
# Description:

install.packages("statmod")   # if needed
install.packages("tweedie")
install.packages("fastDummies")

# Load libraries
library(statmod)
library(tweedie)
library(tidyverse)
library(tidyr)
library(rpart)

library(ggplot2)
library(dplyr)
library(MASS)

library(fastDummies)

library(rpart.plot)
library(randomForest)
library(corrplot)
library(survey)
library(car)

library(caret)
library(Metrics)

# Function that applies the Central Limit Theorem to Standardize the data
standardize <- function(x) {
  if (is.data.frame(x) || is.matrix(x)) {
    # scale all numeric columns
    num_cols <- sapply(x, is.numeric)
    x[num_cols] <- lapply(x[num_cols], function(col) {
      (col - mean(col, na.rm = TRUE)) / sd(col, na.rm = TRUE)
    })
    return(x)
  } else {
    # single vector
    return((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
  }
}


set.seed(123)

n <- 5000

data <- data.frame(
  age = round(rnorm(n, 45, 12)),
  vehicle_age = round(runif(n, 0, 15)),
  region = sample(c("North", "South", "East", "West"), n, replace = TRUE),
  gender = sample(c("M", "F"), n, replace = TRUE),
  exposure = runif(n, 0.5, 1)
)

# Create frequency (Poisson)
lambda <- exp(-3 + 0.02 * data$age + 0.03 * data$vehicle_age)
data$claim_count <- rpois(n, lambda * data$exposure)

# Create severity (Gamma)
severity <- rgamma(n, shape = 2, scale = 2000)

# Total loss
data$loss <- data$claim_count * severity

head(data)


data$age[sample(1:n, 200)] <- NA
data$region[sample(1:n, 150)] <- NA


## DATA CLEANING
# Numeric → median imputation
data$age[is.na(data$age)] <- median(data$age, na.rm = TRUE)

# Categorical → mode imputation
mode_region <- names(sort(table(data$region), decreasing = TRUE))[1]
data$region[is.na(data$region)] <- mode_region

## ENCODE CATEGORICAL VARIABLES
data$region <- as.factor(data$region)
data$gender <- as.factor(data$gender)

## TRAIN/TEST SPLIT
set.seed(123)

train_index <- createDataPartition(data, p = 0.8, list = FALSE)

train <- data[train_index, ]
test  <- data[-train_index, ]


## STANDARDIZATION (ML PIPELINE)
preproc <- preProcess(train[, c("age", "vehicle_age")], method = c("center", "scale"))

train_scaled <- train
test_scaled  <- test

train_scaled[, c("age", "vehicle_age")] <- predict(preproc, train[, c("age", "vehicle_age")])
test_scaled[, c("age", "vehicle_age")]  <- predict(preproc, test[, c("age", "vehicle_age")])

## CROSS-VALIDATION
train_control <- trainControl(
  method = "cv",
  number = 5
)


## FIT TWEEDIE GLM
tweedie_model <- train(
  loss ~ age + vehicle_age + region + gender,
  data = train_scaled,
  method = "glm",
  family = tweedie(var.power = 1.5, link.power = 0),  # log link
  trControl = train_control
)


## MODEL SUMMARY
summary(tweedie_model$finalModel)


## PREDICTIONS
preds <- predict(tweedie_model, newdata = test_scaled)


## RMSE & MAE
rmse_val <- rmse(test_scaled$loss, preds)
mae_val  <- mae(test_scaled$loss, preds)

rmse_val
mae_val


## Compute Actual vs Predicted
results <- data.frame(
  actual = test_scaled$loss,
  predicted = preds
)

ggplot(results, aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.3) +
  geom_abline(color = "red") +
  ggtitle("Actual vs Predicted Loss")




