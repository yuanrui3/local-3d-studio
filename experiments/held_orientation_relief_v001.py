"""Synthetic registered torso patch for held-orientation relief versus albedo."""
import argparse
import json
from pathlib import Path

import numpy as np

FIT_YAW = (-32., 0., 32.)
HELD_YAW = (-46., 48.)
LOW_YAW = ((-3., 0., 3.), (-5., 5.))
GRID_MM = np.arange(-6., 6.01, .5)
TRUTH_MM = 3.6
AMBIENT = .25
DIFFUSE = .75
LIGHT = np.array([.54, .22, .81], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)
LIMITS = {"minimum_fit_yaw_span_deg": 45., "held_yaw_beyond_fit_deg": 8.,
          "max_light_reference_delta": .02, "max_cross_polar_excess_p95": .025,
          "max_registration_fiducial_px": .5, "max_normal_proof_angle_deg": .01,
          "max_amplitude_error_mm": 1., "max_held_relief_to_albedo_RMSE": .55,
          "max_fake_held_RMSE": 1e-8, "min_true_profile_shift_mm": 3., "max_fake_profile_shift_mm": 1e-8}


def unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def patch_grid():
    return np.meshgrid(np.linspace(-.08, .08, 65), np.linspace(-.06, .06, 49))


def bump(u, v, u0=0., v0=0., su=.015, sv=.038):
    x, y = (u-u0)/su, (v-v0)/sv
    value = np.exp(-.5*(x*x+y*y))
    return value, -x/su*value, -y/sv*value


def smooth_height(u, v):
    return -.5*(u*u/.33+v*v/.65)


def height(u, v, amplitude_mm, *, out_of_basis=False):
    result = smooth_height(u, v) + amplitude_mm * .001 * bump(u, v)[0]
    if out_of_basis:
        result += .00024 * bump(u, v, .018, -.012, .011, .016)[0]
    return result


def analytic_normals(u, v, amplitude_mm, *, out_of_basis=False):
    _, dx, dy = bump(u, v)
    du, dv = -u/.33 + amplitude_mm*.001*dx, -v/.65 + amplitude_mm*.001*dy
    if out_of_basis:
        _, dx2, dy2 = bump(u, v, .018, -.012, .011, .016)
        du, dv = du+.00024*dx2, dv+.00024*dy2
    return unit(np.stack((-du, -dv, np.ones_like(u)), axis=-1))


def truth_normals(u, v, amplitude_mm, *, out_of_basis=False):
    """Generate fixture normals from sampled heights, independently of analytic gradients."""
    h = 1.e-5
    def z(x, y):
        return height(x, y, amplitude_mm, out_of_basis=out_of_basis)
    dx = (z(u+h, v)-z(u-h, v))/(2*h)
    dy = (z(u, v+h)-z(u, v-h))/(2*h)
    return unit(np.stack((-dx, -dy, np.ones_like(dx)), axis=-1))


def normal_angles(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1)))


def rotate_y(normals, yaw):
    rad = np.radians(yaw)
    c, s = np.cos(rad), np.sin(rad)
    x, y, z = np.moveaxis(normals, -1, 0)
    return np.stack((c*x+s*z, y, -s*x+c*z), axis=-1)


def shading(normals, yaw, light=LIGHT):
    return AMBIENT+DIFFUSE*np.maximum(rotate_y(normals, yaw) @ light, 0)


def smooth_lowpass(value):
    kernel = np.array([1., 4., 6., 4., 1.])/16
    for axis in (-2, -1):
        pad = [(0, 0)]*value.ndim
        pad[axis] = (2, 2)
        padded = np.pad(value, pad, mode="edge")
        n = value.shape[axis]
        slices = np.moveaxis(padded, axis, -1)
        value = np.moveaxis(sum(k*slices[..., i:i+n] for i, k in enumerate(kernel)), -1, axis)
    return value


def calibration_luma(light):
    panels = unit(np.array([[0., 0., 1.], [-.48, 0., .88], [0., .54, .84]]))
    return .7*shading(panels, 0., light)


def observation_block(u, v, yaw, scenario, *, held):
    base = truth_normals(u, v, 0.)
    relief = truth_normals(u, v, TRUTH_MM, out_of_basis=True)
    albedo = .63 + .045*np.cos(22*u+11*v)
    if scenario == "fake_albedo":
        # Match the true fold exactly at a single orientation with a static
        # material field, then keep that material fixed at every other view.
        albedo = albedo*shading(relief, 0.)/shading(base, 0.)
        surface = base
    else:
        surface = relief
    nominal = calibration_luma(LIGHT)
    target_fiducials = np.array([[12., 11.], [51., 12.], [14., 36.], [50., 37.]])
    images, references, polar_excess, fiducials = [], [], [], []
    for index, angle in enumerate(yaw):
        light = unit(LIGHT+np.array([-.32, .10, .02])) if scenario == "light_drift" and held and index == 0 else LIGHT
        cross = albedo*shading(surface, angle, light)
        excess = .14*bump(u, v, .008, -.004, .025, .030)[0] if scenario == "specular" and index == 0 else np.zeros_like(u)
        regular = cross+excess
        offset = np.array([1.50, -.75]) if scenario == "registration" and held and index == 0 else np.zeros(2)
        images.append(regular)
        references.append(calibration_luma(light)-nominal)
        polar_excess.append(regular-cross)
        fiducials.append((target_fiducials+offset)-target_fiducials)
    return {"yaw": tuple(yaw), "intensity": np.stack(images),
            "reference_delta": np.stack(references), "polar_excess": np.stack(polar_excess),
            "fiducial_residual": np.stack(fiducials)}


def quality(fit, held):
    f, h = fit["yaw"], held["yaw"]
    light = float(max(np.max(np.abs(group["reference_delta"])) for group in (fit, held)))
    polar = float(max(np.max(np.quantile(group["polar_excess"], .95, axis=(-2, -1))) for group in (fit, held)))
    registration = float(max(np.max(np.linalg.norm(group["fiducial_residual"], axis=-1)) for group in (fit, held)))
    yaw_span = max(f)-min(f)
    held_distance = min(min(f)-min(h), max(h)-max(f))
    blocked = []
    if yaw_span < LIMITS["minimum_fit_yaw_span_deg"] or held_distance < LIMITS["held_yaw_beyond_fit_deg"]:
        blocked.append("low_yaw")
    if light > LIMITS["max_light_reference_delta"]:
        blocked.append("light_drift")
    if polar > LIMITS["max_cross_polar_excess_p95"]:
        blocked.append("specular")
    if registration > LIMITS["max_registration_fiducial_px"]:
        blocked.append("registration")
    return {"blocked": blocked, "fit_yaw_span_deg": yaw_span, "held_yaw_beyond_fit_deg": held_distance,
            "max_light_reference_delta": light, "max_cross_polar_excess_p95": polar,
            "max_registration_fiducial_px": registration}


def ratio(u, v, block):
    base = analytic_normals(u, v, 0.)
    denominator = smooth_lowpass(np.stack([shading(base, yaw) for yaw in block["yaw"]]))
    return block["intensity"]/denominator, denominator


def pattern(u, v, yaw, amplitude, denominator):
    normals = analytic_normals(u, v, amplitude)
    return np.stack([shading(normals, angle) for angle in yaw])/denominator


def fit_hypotheses(u, v, fit):
    observed, denominator = ratio(u, v, fit)
    fitted = []
    for amplitude in GRID_MM:
        form = pattern(u, v, fit["yaw"], amplitude, denominator)
        albedo = np.sum(form*observed, axis=0)/np.sum(form*form, axis=0)
        loss = float(np.sqrt(np.mean((albedo*form-observed)**2)))
        fitted.append((float(amplitude), albedo, loss))
    static = next(row for row in fitted if row[0] == 0)
    relief = min(fitted, key=lambda row: (row[2], abs(row[0]), row[0]))
    return static, relief


def score_held(u, v, held, hypotheses):
    observed, denominator = ratio(u, v, held)
    result = {}
    for key, (amplitude, albedo, fit_rmse) in zip(("static_albedo", "relief_and_albedo"), hypotheses, strict=True):
        error = pattern(u, v, held["yaw"], amplitude, denominator)*albedo-observed
        result[key] = {"amplitude_mm": amplitude, "fit_RMSE": fit_rmse,
                       "held_RMSE": float(np.sqrt(np.mean(error*error))),
                       "held_RMSE_by_yaw": [float(value) for value in np.sqrt(np.mean(error*error, axis=(-2, -1)))]}
    return result


def profile_shift_mm(amplitude, *, extra=False):
    # At yaw +90 degrees, the orthographic horizontal support is the maximum
    # world height on the centre row. This uses geometry alone, never shading.
    u = np.linspace(-.08, .08, 801)
    v = np.zeros_like(u)
    return 1000*float(height(u, v, amplitude, out_of_basis=extra).max()-smooth_height(u, v).max())


def compare():
    u, v = patch_grid()
    truth = truth_normals(u, v, TRUTH_MM, out_of_basis=True)
    proof = float(np.max(normal_angles(truth, analytic_normals(u, v, TRUTH_MM, out_of_basis=True))))
    positive_ref = observation_block(u, v, (0.,), "genuine_relief", held=False)["intensity"]
    fake_ref = observation_block(u, v, (0.,), "fake_albedo", held=False)["intensity"]
    match = float(np.max(np.abs(positive_ref-fake_ref)))
    outcomes = {}
    for scenario in ("genuine_relief", "fake_albedo", "light_drift", "specular", "low_yaw", "registration"):
        fit_yaw, held_yaw = LOW_YAW if scenario == "low_yaw" else (FIT_YAW, HELD_YAW)
        fit = observation_block(u, v, fit_yaw, scenario, held=False)
        held = observation_block(u, v, held_yaw, scenario, held=True)
        gate = quality(fit, held)
        if gate["blocked"]:
            outcomes[scenario] = {"status": "abstained", "fit_performed": False, "quality": gate}
            continue
        hypotheses = fit_hypotheses(u, v, fit)
        scores = score_held(u, v, held, hypotheses)
        amplitude = scores["relief_and_albedo"]["amplitude_mm"]
        actual = TRUTH_MM if scenario == "genuine_relief" else 0.
        outcomes[scenario] = {"status": "scored", "fit_performed": True, "quality": gate, "scores": scores,
                              "geometry_only_profile_truth_shift_mm": profile_shift_mm(actual, extra=actual != 0),
                              "geometry_only_profile_fitted_shift_mm": profile_shift_mm(amplitude)}
    positive, fake = outcomes["genuine_relief"], outcomes["fake_albedo"]
    genuine_fit, genuine_base = [positive["scores"][key] for key in ("relief_and_albedo", "static_albedo")]
    fake_fit = fake["scores"]["relief_and_albedo"]
    predicted = analytic_normals(u, v, genuine_fit["amplitude_mm"])
    normal_rmse = float(np.sqrt(np.mean(normal_angles(truth, predicted)**2)))
    null_normal_rmse = float(np.sqrt(np.mean(normal_angles(truth, analytic_normals(u, v, 0.))**2)))
    conditions = {
        "known_normals": proof <= LIMITS["max_normal_proof_angle_deg"],
        "single_view_fake_matches_true": match < 1e-12,
        "genuine_amplitude": abs(genuine_fit["amplitude_mm"]-TRUTH_MM) <= LIMITS["max_amplitude_error_mm"],
        "genuine_held_gain": genuine_fit["held_RMSE"] <= LIMITS["max_held_relief_to_albedo_RMSE"]*genuine_base["held_RMSE"],
        "fake_stays_albedo": fake_fit["amplitude_mm"] == 0 and fake_fit["held_RMSE"] <= LIMITS["max_fake_held_RMSE"],
        "independent_geometry_profile": positive["geometry_only_profile_truth_shift_mm"] >= LIMITS["min_true_profile_shift_mm"]
            and abs(fake["geometry_only_profile_truth_shift_mm"]) <= LIMITS["max_fake_profile_shift_mm"],
        "confounds_abstain": all(outcomes[case]["status"] == "abstained" and outcomes[case]["quality"]["blocked"] == [case]
                                 for case in ("light_drift", "specular", "low_yaw", "registration")),
    }
    return {"type": "registered-torso-held-orientation-relief-v001", "schema_version": 1, "passed": all(conditions.values()),
            "conditions": conditions, "fit_yaw_degrees": FIT_YAW, "held_yaw_degrees": HELD_YAW,
            "amp_grid_mm": GRID_MM.tolist(), "true_main_amplitude_mm": TRUTH_MM, "limits": LIMITS,
            "known_normal_max_error_deg": proof, "predicted_normal_RMSE_deg": normal_rmse,
            "smooth_null_normal_RMSE_deg": null_normal_rmse, "zero_yaw_pair_max_intensity_difference": match,
            "outcomes": outcomes, "source_media_or_model_used": False, "GPU_used": False,
            "scope": "Generated material-space registration, fixed known smooth base and world light, one relief basis plus a held-out truth lobe. "
                     "The extra light, polar and registration controls are generated. This does not validate anatomy or apply uncalibrated video."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "conditions": result["conditions"]}))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
