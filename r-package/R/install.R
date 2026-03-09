#' Install the grecov Python package
#'
#' Helper to install the grecov Python package into the reticulate
#' Python environment.
#'
#' @param method Installation method passed to [reticulate::py_install()].
#' @param ... Additional arguments passed to [reticulate::py_install()].
#'
#' @export
install_grecov <- function(method = "auto", ...) {
  reticulate::py_install("grecov", method = method, ...)
}
