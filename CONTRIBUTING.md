# Contributing to Local 3D Studio

Start with the [README quickstart](README.md) and run the documented smoke test. For a setup problem or a bug, open a GitHub issue with the smallest reproducible example. For a new model, evaluator, or change to the reconstruction pipeline, open an experiment proposal so we can agree on the baseline and test inputs before implementation.

## Set up a change

1. Fork the repository and create a branch from `main`.
2. Install the dependencies for the workflow you plan to change. Use a clean environment for setup and compatibility fixes.
3. Run the affected tests before and after your change. Include the exact commands and results in your pull request.
4. Open a pull request against `main`. Explain what users will see and link the issue if one exists.

Small fixes to installation, error messages, example inputs, or documentation are welcome. Useful research contributions include camera and mask checks, CPU-friendly test fixtures, and better comparisons across reserved views.

## Report reconstruction changes

Compare the current baseline and your change on the same inputs, camera, crop and display settings. For an accuracy claim, report the metric and its units, identify which inputs were used for fitting, and reserve at least one observation for evaluation. Share side-by-side renders when the underlying material can be redistributed. Call out any visible regressions or conditions where the result is uncertain.

Use a synthetic fixture or material you have permission to share. Keep credentials, private footage, local machine paths, model checkpoints, and generated output out of commits and issue attachments. When adding an optional dependency or dataset, link its upstream source and document its license, version, and any separate terms for weights or data.
