# Release Process

## Python (PyPI)

Releases are automated. To publish a new version:

1. Bump the `version` in `pyproject.toml`
2. Push to `main`

The build workflow will detect the new version tag doesn't exist yet, build wheels for all platforms (Linux, macOS, Windows, Pyodide), create a GitHub release, and publish to PyPI.

## R (CRAN)

After the Python release is published:

1. Update the version in `r-package/DESCRIPTION` if needed
2. Go to **Actions > Submit to CRAN** and click **Run workflow**

This runs `R CMD check` and submits the package to CRAN. A GitHub issue is created to track the submission status.
