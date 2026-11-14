# Third-party notices

The root MIT license applies to this project's original code and documentation. The following components keep their upstream terms.

| Component | Role | License or terms |
| --- | --- | --- |
| [Google model-viewer 3.5.0](https://github.com/google/model-viewer/tree/v3.5.0) | Vendored browser viewer | [Apache 2.0](vendor/web/LICENSE-model-viewer), with bundled [Three.js](vendor/web/LICENSE-three), [Lit](vendor/web/LICENSE-lit), [fflate](vendor/web/LICENSE-fflate), and [gainmap-js](vendor/web/LICENSE-gainmap) |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | Downloaded reconstruction source | [MIT](https://github.com/VAST-AI-Research/TripoSR/blob/main/LICENSE) |
| [TripoSR weights](https://huggingface.co/stabilityai/TripoSR) | Downloaded reconstruction checkpoint | MIT, as declared in the upstream model repository |
| [DINO ViT-B/16](https://huggingface.co/facebook/dino-vitb16) | Encoder configuration | Apache 2.0, as declared in the upstream model repository |
| [Segment Anything](https://github.com/facebookresearch/segment-anything) | Optional downloaded mask refinement code and checkpoint | [Apache 2.0](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE) |
| [Torchvision](https://github.com/pytorch/vision) | Optional DeepLabV3, Mask R-CNN and Keypoint R-CNN model weights | [BSD 3-Clause for source code](https://github.com/pytorch/vision/blob/main/LICENSE). See the upstream model and training data terms for weights. |
| [rembg](https://github.com/danielgatis/rembg) | Foreground extraction library | MIT, per its upstream license |
| [U²-Net human segmentation](https://github.com/danielgatis/rembg/releases/tag/v0.0.0) | Downloaded foreground extraction model | See [U²-Net](https://github.com/xuebinqin/U-2-Net) and the checkpoint distributor for applicable terms |

The `vendor/sources.json` and `weights-manifest.json` manifests identify the exact upstream revisions and assets supported by the download commands. Downloaded upstream software and weights live in ignored local directories.

The [real-person demo](docs/real-person-demo.md) identifies the CC0 source photograph and the changes used to produce the preview and mesh.
