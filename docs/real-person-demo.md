# Real-person reconstruction demo

Local 3D Studio reconstructed the displayed mesh from a single photograph of a clothed adult. Both renders show the same reconstructed surface. The third panel rotates the mesh by 90° to show how the model estimated surfaces viewed obliquely or hidden by the pose. This example has no reference scan or second photograph for geometry validation.

[Open the browser demo](https://yuanrui3.github.io/local-3d-studio/viewer.html?model=docs/assets/real-person-demo.glb) · [Download the GLB](assets/real-person-demo.glb)

## Photo and license

Petr Kratochvil photographed *Full Body Fashion Model*. Public Domain Pictures lists the photograph as CC0 1.0 and reports a model release. The comparison isolates the subject against a neutral background and places that crop beside two renders of the locally generated mesh. The viewer asset rotates the exported mesh about its up axis to align the initial camera. [Original photo and release information](https://www.publicdomainpictures.net/en/view-image.php?image=16352) · [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)

## Reproduce locally

Follow the dependency and model setup in the [photo reconstruction guide](photo-reconstruction.md). Download the published 853 × 1280 JPEG from the source page or use the command below. Its SHA-256 is `f5d33e0ce6279f3fbec8aa23f14265ba0e427cab5413c7829f295bd0c9a66bac`.

```sh
curl -fL https://www.publicdomainpictures.net/pictures/20000/velka/full-body-fashion-model.jpg -o fashion-model.jpg
python prepare_references.py fashion-model.jpg --output outputs/demo/prepared
python generate_local.py outputs/demo/prepared/fashion-model-input.png --output outputs/demo/reconstruction --device cpu --resolution 192 --threads 6
```

Run `python -m http.server 8000 --bind 127.0.0.1` from the repo root and open the [local viewer](http://127.0.0.1:8000/viewer.html?model=outputs/demo/reconstruction/model.glb). The prebuilt demo is also available from the viewer’s **Load real-person demo** button. The CPU run at extraction resolution 192 produced a watertight mesh with 9,453 vertices and 18,914 faces. The mesh has arbitrary scale and vertex colors.
