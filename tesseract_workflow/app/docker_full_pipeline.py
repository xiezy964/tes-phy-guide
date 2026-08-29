"""Run the 50-step inverse-design loop through Docker Tesseract images.

The diffusion network remains in the host process; every component boundary is
crossed through a live Tesseract container.  Reverse sensitivities are issued
explicitly through the component VJP endpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tesseract_workflow" / "src"))

import flax.serialization
import jax
import jax.numpy as jnp
from jax import config, random

config.update("jax_enable_x64", True)

from diffusion.quad4_heat_guidance import quad4_loss_and_grad
from diffusion.unet import UNet
from phyguide_tesseract.meshing import bilinear_sample_vjp
from tesseract_core import Tesseract


def load_model(path: Path, key):
    model = UNet()
    dummy = jnp.ones((1, 64, 64, 1), dtype=jnp.float64)
    params = model.init(key, dummy, jnp.ones((1,), dtype=jnp.float64))["params"]
    params = flax.serialization.from_bytes(params, path.read_bytes())
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float64), params)
    return jax.jit(lambda x, t: model.apply({"params": params}, x, t))


def beta_schedule(t):
    return 0.1 + t * (20.0 - 0.1)


def alpha_bar(t):
    return jnp.exp(-0.5 * (0.1 * t + 0.5 * (20.0 - 0.1) * t**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--manufacturing-start", type=int, default=20)
    ap.add_argument("--target", type=float, default=30.0)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "tesseract_workflow/results/docker_full50")
    ap.add_argument("--model", type=Path, default=ROOT / "diffusion/models/vpsde_model_6400.flax")
    args = ap.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"diffusion checkpoint not found: {args.model}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diffusion_dir = args.output_dir / "steps" / "diffusion"
    x0_dir = args.output_dir / "steps" / "x0"
    manufacture_dir = args.output_dir / "steps" / "manufacture"
    mesh_dir = args.output_dir / "steps" / "mesh"
    temperature_dir = args.output_dir / "steps" / "temperature"
    for directory in (diffusion_dir, x0_dir, manufacture_dir, mesh_dir, temperature_dir):
        directory.mkdir(parents=True, exist_ok=True)

    key = random.PRNGKey(args.seed)
    key, model_key, init_key = random.split(key, 3)
    predict_noise = load_model(args.model, model_key)
    x_t = random.normal(init_key, (1, 64, 64, 1), dtype=jnp.float64)
    ts = jnp.linspace(1.0, 1.0e-5, args.steps + 1, dtype=jnp.float64)

    handles = [Tesseract.from_image(name) for name in (
        "tes1_diffusion:latest", "tes2_manufacture:latest", "tes3_mesher:latest", "tes4_fem:latest")]
    for h in handles:
        h.serve()
    diffusion, manufacture, mesher, fem = handles
    hist = {k: [] for k in ("conductance", "loss", "mesh_nodes", "mesh_cells", "gradient_norm", "phase")}
    snapshots = []
    last_grad = np.zeros((64, 64))
    final_geometry = None
    started = time.perf_counter()
    try:
        for step in range(args.steps):
            t = ts[step]
            dt = ts[step] - ts[step + 1]
            eps = predict_noise(x_t, jnp.asarray([t]))
            mean = jnp.maximum(jnp.sqrt(alpha_bar(t)), 1e-6)
            std = jnp.maximum(jnp.sqrt(1.0 - alpha_bar(t)), 1e-6)
            x0 = (x_t - std * eps) / mean
            phi = np.clip(np.asarray(x0[0, :, :, 0]), -1.0, 1.0)
            step_name = f"step_{step + 1:03d}.png"
            np.save(x0_dir / step_name.replace(".png", ".npy"), phi)
            plt.imsave(x0_dir / step_name, np.clip((phi + 1.0) / 2.0, 0.0, 1.0), cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", dpi=300)
            geometry = step >= min(args.manufacturing_start, args.steps)
            if not geometry:
                loss, conductance, grad = quad4_loss_and_grad(jnp.asarray(phi), args.target, beta=8.0)
                last_grad = np.asarray(grad)
                nodes = cells = 0
                phase = "pixel-warmup"
            else:
                manufacturing_inputs = {
                    "level_set": phi,
                    "filter_radius": 1,
                    "projection_beta": 8.0,
                    "min_component_size": 4,
                    "keep_largest_component": True,
                    "supersample": 4,
                    "minimum_feature_size": 2.0,
                    "boundary_smoothing": 0.5,
                }
                m = manufacture.apply(manufacturing_inputs)
                clean = np.asarray(m["clean_level_set"])
                q = mesher.apply({"level_set": clean, "background_stride": 16,
                                  "gmsh_interface_scale": 12.0})
                points = np.asarray(q["points"])
                cells = np.asarray(q["cells"], dtype=np.int32)
                tags = np.asarray(q["phase_tags"], dtype=np.int32)
                tri = points[cells]
                twice_area = np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
                valid = np.isfinite(twice_area) & (twice_area > 1.0e-12)
                if not np.all(valid):
                    raise FloatingPointError("Gmsh returned a degenerate TRI3 element")
                top = np.isclose(points[:, 1], 0.0).astype(float)
                bottom = np.isclose(points[:, 1], 1.0).astype(float)
                if not top.any() or not bottom.any():
                    raise ValueError("Gmsh mesh does not contain both thermal boundaries")
                cent = points[cells].mean(axis=1)
                # Match the validated local pipeline exactly: strict two-phase
                # conductivity in the forward solve, with a straight-through
                # material derivative in the reverse pass.
                conductivity = 1.0 + 99.0 * tags
                fin = fem.apply({"points": points, "cells": cells, "conductivity": conductivity,
                                 "top_mask": top, "bottom_mask": bottom})
                conductance = float(np.asarray(fin["conductance"]))
                temperature = np.asarray(fin["temperature"])
                if not np.isfinite(conductance) or not np.isfinite(temperature).all():
                    raise FloatingPointError("TRI3 FEM returned non-finite values")
                plt.imsave(
                    manufacture_dir / step_name, np.asarray(m["binary_mask"]),
                    cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower",
                )
                mesh_figure, mesh_axis = plt.subplots(figsize=(6, 6))
                mesh_axis.tripcolor(
                    points[:, 0], points[:, 1], cells,
                    facecolors=tags, cmap="gray_r", edgecolors="0.58",
                    linewidth=0.12, vmin=0, vmax=1,
                )
                mesh_axis.set_aspect("equal"); mesh_axis.set_xlim(0, 1); mesh_axis.set_ylim(0, 1)
                mesh_axis.set_title(f"Step {step + 1}: {len(points)} nodes / {len(cells)} cells")
                mesh_axis.set_xticks([]); mesh_axis.set_yticks([])
                mesh_figure.tight_layout()
                mesh_figure.savefig(mesh_dir / step_name, dpi=150, bbox_inches="tight")
                plt.close(mesh_figure)
                temp_figure, temp_axis = plt.subplots(figsize=(6, 6))
                temp_plot = temp_axis.tripcolor(
                    points[:, 0], points[:, 1], cells, temperature,
                    shading="gouraud", cmap="inferno",
                )
                temp_axis.set_aspect("equal"); temp_axis.set_xlim(0, 1); temp_axis.set_ylim(0, 1)
                temp_axis.set_title(f"Step {step + 1}: FEM temperature")
                temp_axis.set_xticks([]); temp_axis.set_yticks([])
                temp_figure.colorbar(temp_plot, ax=temp_axis, fraction=0.046, pad=0.04)
                temp_figure.tight_layout()
                temp_figure.savefig(temperature_dir / step_name, dpi=150, bbox_inches="tight")
                plt.close(temp_figure)
                loss = (conductance - args.target) ** 2
                cot = {"conductance": 2.0 * (conductance - args.target)}
                fg = fem.vector_jacobian_product({"points": points, "cells": cells, "conductivity": conductivity,
                    "top_mask": top, "bottom_mask": bottom}, ["points", "conductivity"], ["conductance"], cot)
                clean_grad = bilinear_sample_vjp(
                    clean.shape, cent, 99.0 * np.asarray(fg["conductivity"])
                )
                # Gmsh point-motion VJP exposed as a level-set VJP.
                mg = mesher.vector_jacobian_product({"level_set": clean, "background_stride": 16,
                    "gmsh_interface_scale": 12.0}, ["level_set"], ["points"],
                    {"points": np.asarray(fg["points"])})
                clean_grad += np.asarray(mg["level_set"])
                vg = manufacture.vector_jacobian_product(manufacturing_inputs, ["level_set"], ["clean_level_set"],
                    {"clean_level_set": clean_grad})
                last_grad = np.asarray(vg["level_set"])
                final_geometry = (clean, np.asarray(m["binary_mask"]), points, cells, tags, temperature)
                nodes, cells_count = len(points), len(cells)
                phase = "docker-manufacture+gmsh+fem"
            grad_xt = jnp.asarray(last_grad)[None, :, :, None] / mean
            norm = jnp.linalg.norm(grad_xt)
            bt = beta_schedule(t)
            key, nk = random.split(key)
            # Invoke the diffusion container for the actual component boundary.
            d = diffusion.apply({"x_t": np.asarray(x_t), "eps_pred": np.asarray(eps), "physical_gradient": np.asarray(grad_xt),
                                 "beta_t": float(bt), "std": float(std), "mean_coef": float(mean), "dt": float(dt),
                                 "guidance_strength": float(2.0 + 48.0 * (1.0 - t)),
                                 "noise": np.asarray(random.normal(nk, x_t.shape, dtype=jnp.float64))})
            x_t = jnp.asarray(d["x_next"])
            sampled_state = np.asarray(x_t[0, :, :, 0])
            np.save(diffusion_dir / step_name.replace(".png", ".npy"), sampled_state)
            scale = max(float(np.percentile(np.abs(sampled_state), 99.0)), 1.0e-8)
            plt.imsave(diffusion_dir / step_name, np.clip(0.5 + 0.5 * sampled_state / scale, 0.0, 1.0), cmap="gray_r", vmin=0.0, vmax=1.0, origin="lower", dpi=300)
            hist["conductance"].append(float(conductance)); hist["loss"].append(float(loss)); hist["gradient_norm"].append(float(norm)); hist["mesh_nodes"].append(float(nodes)); hist["mesh_cells"].append(float(cells_count if geometry else 0)); hist["phase"].append(int(geometry))
            if step in (0, args.manufacturing_start - 1, args.steps - 1): snapshots.append(np.asarray(x0[0, :, :, 0]))
            print(f"step={step+1:02d}/{args.steps} phase={phase} conductance={float(conductance):.5f} loss={float(loss):.5f} mesh={int(nodes)}/{int(hist['mesh_cells'][-1])}", flush=True)
    finally:
        for h in handles:
            h.teardown()

    final_x = np.asarray(x_t[0, :, :, 0]); final_cont = np.clip((final_x + 1) / 2, 0, 1)
    extra = {}
    if final_geometry is not None:
        extra = dict(manufactured_level_set=final_geometry[0], binary_mask=final_geometry[1],
                     mesh_points=final_geometry[2], final_mesh_cells=final_geometry[3],
                     phase_tags=final_geometry[4], temperature=final_geometry[5])
    np.savez_compressed(args.output_dir / "docker_run_data.npz", final_x=final_x, final_continuous=final_cont, snapshots=np.asarray(snapshots), **extra, **{k: np.asarray(v) for k,v in hist.items()})
    fig, ax = plt.subplots(2, 3, figsize=(15, 9)); ax[0, 0].imshow(final_cont, cmap="gray_r", origin="lower"); ax[0, 0].set_title("Docker diffusion output")
    if final_geometry is not None:
        clean, binary, points, cells_arr, tags, temperature = final_geometry
        ax[0, 1].imshow(binary, cmap="gray_r", origin="lower"); ax[0, 1].set_title("Manufactured binary")
        ax[0, 2].tripcolor(points[:, 0], points[:, 1], cells_arr, facecolors=tags, cmap="gray_r", edgecolors="0.6", linewidth=0.1); ax[0, 2].set_title(f"Mesh ({len(points)} / {len(cells_arr)})")
        ax[1, 0].tripcolor(points[:, 0], points[:, 1], cells_arr, temperature, shading="gouraud", cmap="inferno"); ax[1, 0].set_title("FEM temperature")
    else:
        ax[0, 1].axis("off"); ax[0, 2].axis("off"); ax[1, 0].axis("off")
    ax[1, 1].plot(hist["conductance"]); ax[1, 1].axhline(args.target, ls="--", c="r"); ax[1, 1].set_title("Conductance")
    ax[1, 2].plot(hist["loss"]); ax[1, 2].set_yscale("log"); ax[1, 2].set_title("Loss"); fig.tight_layout(); fig.savefig(args.output_dir / "docker_summary.png", dpi=180); plt.close(fig)
    (args.output_dir / "metrics.json").write_text(json.dumps({"steps": args.steps, "seed": args.seed, "target": args.target, "total_seconds": time.perf_counter()-started, "final_conductance": hist["conductance"][-1]}, indent=2))
    print(f"results={args.output_dir}")


if __name__ == "__main__":
    main()
