"""Regenerate the viewer's analytic surface example with NumPy."""

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments import held_orientation_relief_v001 as fixture


def main():
    u, v = fixture.patch_grid()
    z = fixture.height(u, v, fixture.TRUTH_MM, out_of_basis=True)
    vertices = np.stack((u, v, z), axis=-1).reshape(-1, 3).astype("<f4")
    normals = fixture.analytic_normals(u, v, fixture.TRUTH_MM, out_of_basis=True)
    normals = normals.reshape(-1, 3).astype("<f4")
    grid = np.arange(u.size, dtype="<u2").reshape(u.shape)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    indices = np.stack((a, b, d, a, d, c), axis=-1).astype("<u2").ravel()
    arrays = (vertices, normals, indices)
    views = []
    offset = 0
    for values, target in zip(arrays, (34962, 34962, 34963), strict=True):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": values.nbytes, "target": target})
        offset += values.nbytes
    definition = {
        "asset": {"version": "2.0", "generator": "Local 3D Studio synthetic fixture"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"name": "Synthetic raised fold", "primitives": [
            {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}
        ]}],
        "materials": [{"doubleSided": True, "pbrMetallicRoughness": {
            "baseColorFactor": [.16, .38, .30, 1.], "metallicFactor": 0., "roughnessFactor": .74
        }}],
        "buffers": [{"byteLength": offset}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "type": "VEC3", "count": len(vertices),
             "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "type": "VEC3", "count": len(normals)},
            {"bufferView": 2, "componentType": 5123, "type": "SCALAR", "count": len(indices)},
        ],
    }
    manifest = json.dumps(definition, separators=(",", ":")).encode("ascii")
    manifest += b" " * (-len(manifest) % 4)
    data = b"".join(values.tobytes() for values in arrays)
    data += b"\x00" * (-len(data) % 4)
    header = struct.pack("<4sII", b"glTF", 2, 28 + len(manifest) + len(data))
    manifest_chunk = struct.pack("<I4s", len(manifest), b"JSON") + manifest
    data_chunk = struct.pack("<I4s", len(data), b"BIN\x00") + data
    Path(__file__).with_name("synthetic-patch.glb").write_bytes(header + manifest_chunk + data_chunk)


if __name__ == "__main__":
    main()
