# ============================================================
#  GLM from scratch in R
#  Supports: gaussian (identity), binomial (logit), poisson (log)
# ============================================================

# --- 1. Family objects: link, inverse-link, variance, d(mu)/d(eta) ----

make_family <- function(family = c("gaussian", "binomial", "poisson")) {
  family <- match.arg(family)
  
  switch(family,
         gaussian = list(
           name     = "gaussian",
           link     = function(mu) mu,                      # g(mu)   = mu
           linkinv  = function(eta) eta,                    # g^-1    = eta
           mu.eta   = function(eta) rep(1, length(eta)),    # dmu/deta
           variance = function(mu) rep(1, length(mu))       # V(mu)
         ),
         binomial = list(
           name     = "binomial",
           link     = function(mu) log(mu / (1 - mu)),      # logit
           linkinv  = function(eta) 1 / (1 + exp(-eta)),    # sigmoid
           mu.eta   = function(eta) {
             p <- 1 / (1 + exp(-eta))
             p * (1 - p)
           },
           variance = function(mu) mu * (1 - mu)
         ),
         poisson = list(
           name     = "poisson",
           link     = function(mu) log(mu),                 # log
           linkinv  = function(eta) exp(eta),               # exp
           mu.eta   = function(eta) exp(eta),
           variance = function(mu) mu
         )
  )
}


# --- 2. IRLS solver -----------------------------------------------

glm_fit <- function(X, y, family, weights = NULL,
                    maxit = 100, tol = 1e-8) {
  
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(weights)) weights <- rep(1, n)
  
  # Initialize mu (avoid 0/1 for binomial)
  mu <- switch(family$name,
               gaussian = y,
               binomial = pmax(pmin(y, 0.99), 0.01),
               poisson  = pmax(y, 0.1)
  )
  
  eta  <- family$link(mu)
  beta <- rep(0, p)
  
  for (iter in seq_len(maxit)) {
    beta_old <- beta
    
    # Working weights and working response
    dmu_deta <- family$mu.eta(eta)
    V        <- family$variance(mu)
    
    W <- as.vector(weights * dmu_deta^2 / V)   # IRLS weight
    z <- eta + (y - mu) / dmu_deta              # working response
    
    # Weighted least squares: beta = (X'WX)^{-1} X'Wz
    W_sqrt <- sqrt(W)
    Xw     <- X * W_sqrt          # scale rows
    zw     <- z * W_sqrt
    
    beta   <- solve(t(Xw) %*% Xw, t(Xw) %*% zw)
    beta   <- as.vector(beta)
    
    # Update linear predictor and mean
    eta <- as.vector(X %*% beta)
    mu  <- family$linkinv(eta)
    
    # Convergence check
    if (max(abs(beta - beta_old)) < tol) {
      message(sprintf("Converged in %d iterations", iter))
      break
    }
    if (iter == maxit)
      warning("IRLS did not converge — consider increasing maxit")
  }
  
  list(
    coefficients = beta,
    fitted.values = mu,
    linear.predictors = eta,
    family = family,
    n_iter = iter
  )
}


# --- 3. High-level wrapper matching glm() interface ---------------

my_glm <- function(formula, data, family = "gaussian",
                   weights = NULL, maxit = 100, tol = 1e-8) {
  
  fam  <- make_family(family)
  mf   <- model.frame(formula, data)
  y    <- model.response(mf)
  X    <- model.matrix(formula, data)          # includes intercept
  
  fit  <- glm_fit(X, y, fam,
                  weights = weights,
                  maxit   = maxit,
                  tol     = tol)
  
  fit$call    <- match.call()
  fit$formula <- formula
  fit$data    <- data
  class(fit)  <- "my_glm"
  fit
}


# --- 4. S3 helpers ------------------------------------------------

print.my_glm <- function(x, ...) {
  cat("my_glm — family:", x$family$name, "\n\n")
  cat("Coefficients:\n")
  print(x$coefficients)
  invisible(x)
}

predict.my_glm <- function(object, newdata = NULL,
                           type = c("link", "response"), ...) {
  type <- match.arg(type)
  if (is.null(newdata)) {
    eta <- object$linear.predictors
  } else {
    X   <- model.matrix(object$formula, newdata)
    eta <- as.vector(X %*% object$coefficients)
  }
  if (type == "link")     return(eta)
  if (type == "response") return(object$family$linkinv(eta))
}

coef.my_glm <- function(object, ...) object$coefficients


# ============================================================
#  Examples
# ============================================================

# --- Gaussian (equivalent to lm) -----
set.seed(42)
n   <- 200
dat <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n)
)
dat$y <- 3 + 1.5 * dat$x1 - 0.8 * dat$x2 + rnorm(n)

fit_gauss  <- my_glm(y ~ x1 + x2, data = dat, family = "gaussian")
fit_check  <- glm(y ~ x1 + x2,   data = dat, family = gaussian())

cat("\n--- Gaussian ---\n")
cat("My GLM:  "); print(round(coef(fit_gauss), 4))
cat("Base glm:"); print(round(coef(fit_check), 4))


# --- Binomial (logistic regression) -----
dat$p  <- 1 / (1 + exp(-(0.5 + 2 * dat$x1 - dat$x2)))
dat$yb <- rbinom(n, 1, dat$p)

fit_bin   <- my_glm(yb ~ x1 + x2, data = dat, family = "binomial")
fit_binR  <- glm(yb ~ x1 + x2,   data = dat, family = binomial())

cat("\n--- Binomial ---\n")
cat("My GLM:  "); print(round(coef(fit_bin), 4))
cat("Base glm:"); print(round(coef(fit_binR), 4))


# --- Poisson (count regression) -----
dat$yp <- rpois(n, exp(0.3 + 0.7 * dat$x1))

fit_pois  <- my_glm(yp ~ x1 + x2, data = dat, family = "poisson")
fit_poisR <- glm(yp ~ x1 + x2,   data = dat, family = poisson())

cat("\n--- Poisson ---\n")
cat("My GLM:  "); print(round(coef(fit_pois), 4))
cat("Base glm:"); print(round(coef(fit_poisR), 4))


# --- Prediction -----
new_obs <- data.frame(x1 = c(0.5, -1.0), x2 = c(1.0, 0.2))
predict(fit_bin, newdata = new_obs, type = "response")



