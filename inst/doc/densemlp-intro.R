## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(collapse = TRUE, comment = "#>")

## ----setup--------------------------------------------------------------------
library(densemlp)

## -----------------------------------------------------------------------------
set.seed(1)
n <- 300
x <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n))
y <- 2 * x$a - x$b + 0.5 * x$a * x$c + rnorm(n, sd = 0.2)

train <- sample.int(n, floor(0.8 * n))
test <- setdiff(seq_len(n), train)

fit <- densemlp(x[train, ], y[train], hidden_units = c(32, 16), epochs = 60)
pred <- predict(fit, x[test, ])
sqrt(mean((pred - y[test])^2))

## -----------------------------------------------------------------------------
y_class <- factor(ifelse(y > median(y), "high", "low"))
fit_class <- densemlp(x[train, ], y_class[train], hidden_units = c(32, 16), epochs = 60)
predict(fit_class, x[test, ], type = "prob")[1:5, ]
predict(fit_class, x[test, ], type = "class")[1:5]

## -----------------------------------------------------------------------------
fit_opts <- densemlp(
  x[train, ], y[train], hidden_units = c(32, 16),
  residual = TRUE,          # learned skip connection per hidden block
  gated = TRUE,              # learned sigmoid gate per hidden block
  dropout = 0.05,            # inverted dropout per hidden layer
  batch_norm = TRUE,         # batch normalization inside each hidden block
  input_projection = 8,      # linear map of the encoded inputs before layer 1
  ema_decay = 0.99,          # moving-average weights for eval/final predictions
  lr_schedule = "cosine",
  epochs = 60
)

## -----------------------------------------------------------------------------
fit_ens <- densemlp(
  x[train, ], y[train], hidden_units = c(32, 16), epochs = 60,
  ensemble = 5, ncores = 2
)
sqrt(mean((predict(fit_ens, x[test, ]) - y[test])^2))

## -----------------------------------------------------------------------------
cv <- cv_densemlp(x, y, folds = 5, hidden_units = c(32, 16), epochs = 40)
cv

## -----------------------------------------------------------------------------
tuned <- tune_densemlp(
  x, y, repeats = 2,
  grid = list(
    hidden_units = list(c(16), c(32, 16)),
    lr = c(1e-3, 3e-3)
  )
)
tuned$best_config

## ----eval = requireNamespace("survival", quietly = TRUE)----------------------
library(survival)
data(lung, package = "survival")
lung <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog", "ph.karno", "wt.loss")])
sy <- Surv(lung$time, lung$status == 2)
sx <- lung[, c("age", "sex", "ph.ecog", "ph.karno", "wt.loss")]

sfit <- densemlp(sx, sy, task = "survival", hidden_units = c(16, 8), epochs = 60)
risk <- predict(sfit, sx, type = "response")
densemlp_metrics(sy, risk, task = "survival")

## -----------------------------------------------------------------------------
imp <- perm_importance(fit, x[test, ], y[test])
imp$data

## ----fig.width = 6, fig.height = 4--------------------------------------------
plot(fit)          # or, equivalently, plot_history(fit)

