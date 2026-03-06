grecov_mod <- NULL

.onLoad <- function(libname, pkgname) {
  grecov_mod <<- reticulate::import("grecov", delay_load = TRUE)
}
