# Held-view relief benchmark

The benchmark tests whether a change in shading supports a change in 3D surface orientation. It generates a small curved patch with a known raised feature and a second patch whose color reproduces the raised patch’s head-on appearance. Both use known light and material-space registration. The geometry includes an extra lobe outside the fitting basis.

## Fit and evaluate

- Fit a static-color hypothesis and a relief-plus-color hypothesis on yaw angles −32°, 0°, and 32°.
- Score both on the reserved −46° and 48° views. Report signed appearance error, the fitted amplitude, and surface-normal error against the generated reference.
- Compare the height profiles against independent generated geometry. Check that the painted case yields zero fitted relief.
- Run generated controls for light drift, specular reflections, narrow viewpoint coverage, and registration error. Each control should abstain before fitting.

The [README commands](../README.md#run-the-cpu-benchmark) produce a JSON report. `conditions` contains the pass checks, `outcomes` contains case-specific errors and estimates, and `limits` records the thresholds. The geometry, material, and illumination are generated. A passing result establishes behavior in those conditions. Ordinary phone footage may have unknown illumination, uncertain cameras, changing pose, and specular skin or cloth, which this test does not validate.
