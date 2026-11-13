# Reconstruct a clothed photo locally

This guide uses the single-photo TripoSR backend. The input is a local photo with one visible subject. The output is a vertex-colored GLB and PLY with an estimated shape. The rear surface is inferred from the model prior, and the model has arbitrary scale.

## Install and download

Use Git and Python 3.10 or 3.11 on a 64-bit system. Create and activate a virtual environment as shown in the [README](../README.md#run-the-cpu-benchmark). On Windows or Linux, install the matching CPU builds first.

```sh
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-photo.txt
```

On macOS, run only the second command to install the matching packages from PyPI. Check [PyTorch’s install guide](https://pytorch.org/get-started/locally/) for compatible device builds if you want to use a GPU.

Restore the pinned upstream code and download the four baseline assets. Allow roughly 2 GB for weights, plus Python dependencies and working files. This is the only part of the baseline workflow that needs internet access.

```sh
python bootstrap_vendor.py
python bootstrap_vendor.py --check
python fetch_models.py --only triposr-config --only triposr-model --only dino-config --only u2net-human-seg
```

## Prepare and reconstruct

Use a photo you have permission to process. A centered, well-lit photo with the full head, hands, and shoes visible gives the foreground extractor a better starting point. Replace `path/to/front.jpg` with your photo and keep the filename stem consistent in the second command.

```sh
python prepare_references.py path/to/front.jpg --output outputs/prepared
python generate_local.py outputs/prepared/front-input.png --output outputs/reconstruction --device cpu --resolution 128 --threads 4
```

Inspect `outputs/prepared/front-cutout.png` for missing limbs or background regions before interpreting the mesh. The reconstruction writes `model.glb`, `model.ply`, and `manifest.json` in `outputs/reconstruction`. The manifest records the input hash, device, upstream pins, elapsed time, and output geometry counts. It also records the local input path, so review it before sharing.

## Open the viewer

Start the server in the repo root and open [the local viewer](http://127.0.0.1:8000/viewer.html). It loads `outputs/reconstruction/model.glb` by default. You can also open a `.glb` from your computer or set the query parameter `?model=relative/path.glb` for another model in the served folder.

```sh
python -m http.server 8000 --bind 127.0.0.1
```

Drag to orbit and use the front, side, and back buttons to check the silhouette and surfaces. For a checkpoint-free example, open the [synthetic patch](http://127.0.0.1:8000/viewer.html?model=docs/assets/synthetic-patch.glb).

## Diagnose a run

- If the input image is already cropped to a centered square with a neutral background, you can pass it directly to `generate_local.py`.
- Start with `--resolution 64` if you want a faster CPU smoke test. Increase the resolution only after verifying that the cutout and geometry are usable.
- Re-run the same `fetch_models.py` command with `--verify-only` to check the local hashes without using the network. Use `--repair` with the same asset list to replace a failed download.
- For experimental mask refinement, `prepare_references.py --help` lists optional modes. `--semantic` uses the `deeplabv3` asset. `--instance` and `--instance-refine` use `maskrcnn`. `--sam` uses `maskrcnn` and `sam-vit-b`, and can use `keypointrcnn` for additional pose support. Fetch those assets by repeating `--only` for the relevant IDs.

See the [held-view benchmark](held-view-benchmark.md) for a reproducible geometry-versus-color control and the [third-party notices](../THIRD_PARTY_NOTICES.md) for upstream terms.
