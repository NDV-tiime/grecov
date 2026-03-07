test_that("multinomial_ci returns valid CI", {
  result <- multinomial_ci(
    counts = c(10, 10, 20, 60),
    values = c(1, 2, 3, 4),
    alpha = 0.05
  )

  expect_type(result, "list")
  expect_true(result$lower < result$upper)
  expect_length(result$p_lower, 4)
  expect_length(result$p_upper, 4)
  expect_true(result$bfs_calls > 0)
  expect_true(result$bfs_total_states > 0)

  # The observed mean is 3.3, CI should contain it
  expect_true(result$lower < 3.3)
  expect_true(result$upper > 3.3)
})

test_that("multinomial_ci works with method='mass'", {
  result <- multinomial_ci(
    counts = c(10, 10, 20, 60),
    values = c(1, 2, 3, 4),
    alpha = 0.05,
    method = "greedy"
  )

  expect_type(result, "list")
  expect_true(result$lower < result$upper)
  expect_true(result$lower < 3.3)
  expect_true(result$upper > 3.3)
})

test_that("multinomial_ci works with 2 categories", {
  result <- multinomial_ci(
    counts = c(30, 70),
    values = c(0, 1),
    alpha = 0.05
  )

  expect_true(result$lower < 0.7)
  expect_true(result$upper > 0.7)
  expect_length(result$p_lower, 2)
})
