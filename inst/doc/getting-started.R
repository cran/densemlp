## ----setup, include=FALSE-----------------------------------------------------
can_use_torch <- requireNamespace("torch", quietly = TRUE) &&
  isTRUE(torch::torch_is_installed())

knitr::opts_chunk$set(
  echo = TRUE,
  eval = can_use_torch,
  collapse = TRUE,
  comment = "#>",
  fig.width = 6,
  fig.height = 4
)

## -----------------------------------------------------------------------------
library(densemlp)

## ----classification-fit-------------------------------------------------------
fit <- densemlp(
  Species ~ .,
  data = iris,
  epochs = 10,
  patience = 3,
  verbose = FALSE,
  seed = 1
)

fit

## ----classification-predictions-----------------------------------------------
predict(fit, iris[1:5, ], type = "class")
round(predict(fit, iris[1:5, ], type = "prob"), 3)

## ----classification-metrics---------------------------------------------------
pred <- predict(fit, iris, type = "class")
densemlp_metrics(iris$Species, pred)

## ----regression-fit-----------------------------------------------------------
fit_reg <- densemlp(
  mpg ~ disp + hp + wt,
  data = mtcars,
  epochs = 10,
  patience = 3,
  verbose = FALSE,
  seed = 2
)

fit_reg

## ----regression-predictions---------------------------------------------------
pred_reg <- predict(fit_reg, mtcars, type = "response")
head(round(pred_reg, 2))
densemlp_metrics(mtcars$mpg, pred_reg)

## ----plot-history-------------------------------------------------------------
plot_history(fit)

## ----autoplot-history---------------------------------------------------------
ggplot2::autoplot(fit)

## ----tuning-------------------------------------------------------------------
tuned <- tune_densemlp(
  Species ~ .,
  data = iris,
  grid = list(
    hidden_units = list(c(8), c(16, 8)),
    activation = c("relu"),
    dropout = c(0),
    batch_size = c(8),
    lr = c(1e-3),
    epochs = c(10)
  ),
  patience = 3,
  seed = 3,
  verbose = FALSE
)

tuned$results

## ----permutation-importance---------------------------------------------------
importance <- perm_importance(fit, iris[, -5], iris$Species)
importance$data
plot(importance)

