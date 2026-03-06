#' Compute an exact confidence interval for a multinomial weighted sum
#'
#' Computes exact confidence intervals for weighted sums of multinomial
#' categories via Neyman construction with the GreCov algorithm.
#'
#' @param counts Integer vector of observed category counts.
#' @param values Numeric vector of values assigned to each category
#'   (same length as `counts`).
#' @param alpha Significance level (default 0.05 for a 95\% CI).
#' @param method Either `"equal_tail"` (default) or `"mass"`.
#' @param eps_ratio BFS stopping tolerance as fraction of alpha (default 1e-3).
#' @param verbose Verbosity level: 0 = silent, 1 = BFS stats, 2 = optimizer
#'   output.
#' @param optimizer Optimizer backend. `NULL` uses the default (`"ipopt"` or
#'   `"trust-constr"` for equal_tail, `"de"` for mass).
#' @param param Parametrization: `"direct"`, `"reduced"`, or `"logit"`. `NULL`
#'   uses the default for the chosen optimizer.
#' @param pmin Minimum probability bound for direct/reduced parametrizations
#'   (default 1e-8).
#' @param theta_max Bound on theta for softmax parametrization (default 10).
#' @param tie_margin Log-probability margin for near-tie exclusion in mass
#'   method (default 1e-8).
#' @param use_python Force pure-Python BFS instead of C++ extension (default
#'   `FALSE`).
#'
#' @return A list with components:
#' \describe{
#'   \item{lower}{Lower confidence interval endpoint.}
#'   \item{upper}{Upper confidence interval endpoint.}
#'   \item{p_lower}{Probability vector at the lower endpoint.}
#'   \item{p_upper}{Probability vector at the upper endpoint.}
#'   \item{bfs_calls}{Number of BFS function calls.}
#'   \item{bfs_total_states}{Total states explored across all BFS calls.}
#' }
#'
#' @examples
#' \dontrun{
#' result <- confidence_interval(
#'   counts = c(10, 10, 20, 60),
#'   values = c(1, 2, 3, 4),
#'   alpha = 0.05
#' )
#' cat(sprintf("95%% CI: [%.4f, %.4f]\n", result$lower, result$upper))
#' }
#'
#' @export
confidence_interval <- function(counts,
                                values,
                                alpha = 0.05,
                                method = "equal_tail",
                                eps_ratio = 1e-3,
                                verbose = 0L,
                                optimizer = NULL,
                                param = NULL,
                                pmin = 1e-8,
                                theta_max = 10.0,
                                tie_margin = 1e-8,
                                use_python = FALSE) {
  result <- grecov_mod$confidence_interval(
    counts = as.integer(counts),
    values = as.double(values),
    alpha = alpha,
    method = method,
    eps_ratio = eps_ratio,
    verbose = as.integer(verbose),
    optimizer = optimizer,
    param = param,
    pmin = pmin,
    theta_max = theta_max,
    tie_margin = tie_margin,
    use_python = use_python
  )

  list(
    lower = result$lower,
    upper = result$upper,
    p_lower = as.numeric(result$p_lower),
    p_upper = as.numeric(result$p_upper),
    bfs_calls = as.integer(result$bfs_calls),
    bfs_total_states = as.integer(result$bfs_total_states)
  )
}
