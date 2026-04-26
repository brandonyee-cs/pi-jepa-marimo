"""
precompute.py  —  run this once locally, then paste the output into pijepa_toolkit.py
  pip install numpy
  python precompute.py

It will print a line like:
  _DARCY_CACHE_B64 = "ABC123..."
Copy that entire line into pijepa_toolkit.py where marked.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import io, base64
import pijepa_toolkit as T

print("Computing hero cache  (seeds 0–19, channelized, n=32) ...")
hero = {}
for seed in range(20):
    K = T.make_permeability_channelized(32, n_layers=3, seed=seed)
    p = T.solve_darcy_fd(K)
    ux, uy = T.darcy_velocity(K, p)
    ctx, tgt = T.spatiotemporal_block_mask(32, context_frac=0.65, seed=seed)
    hero[f"h_K_{seed}"]   = K
    hero[f"h_p_{seed}"]   = p
    hero[f"h_ux_{seed}"]  = ux
    hero[f"h_uy_{seed}"]  = uy
    hero[f"h_ctx_{seed}"] = ctx.astype(np.uint8)
    hero[f"h_tgt_{seed}"] = tgt.astype(np.uint8)
    print(f"  seed {seed:02d} done")

print("Computing split cache (seeds 0–29, channelized + GRF, n=32) ...")
for seed in range(30):
    for prefix, K in [
        ("sc", T.make_permeability_channelized(32, n_layers=3, seed=seed)),
        ("sg", T.make_permeability_grf(32, length_scale=0.20, variance=2.0, seed=seed)),
    ]:
        p = T.solve_darcy_fd(K)
        ux, uy = T.darcy_velocity(K, p)
        hero[f"{prefix}_K_{seed}"]  = K
        hero[f"{prefix}_p_{seed}"]  = p
        hero[f"{prefix}_ux_{seed}"] = ux
        hero[f"{prefix}_uy_{seed}"] = uy
    print(f"  seed {seed:02d} done")

print("Encoding ...")
buf = io.BytesIO()
np.savez_compressed(buf, **hero)
b64 = base64.b64encode(buf.getvalue()).decode("ascii")

# Write directly to a file so you don't have to copy from terminal
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_b64.txt")
with open(out_path, "w") as f:
    f.write(b64)

print(f"\nDone! Base64 written to: {out_path}")
print(f"Size: {len(b64):,} characters ({len(b64)//1024} KB)")
print("\nNow open pijepa_toolkit.py and replace the empty string in:")
print('  _DARCY_CACHE_B64 = ""')
print("with the contents of cache_b64.txt")
