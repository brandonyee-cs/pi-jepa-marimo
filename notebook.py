# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.2",
#     "numpy>=1.26",
#     "matplotlib>=3.9",
# ]
# ///
"""
PI-JEPA: Closing the Data Asymmetry in Subsurface Surrogate Modeling
─────────────────────────────────────────────────────────────────────
Interactive companion to arXiv:2604.01349
  Brandon Yee · Pairie Koh — Yee Collins Research Group / Hoover Institution at Stanford

No torch. All computation is numpy-only; the notebook loads in seconds.
Paper results are pre-loaded from pijepa_toolkit.py.
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".")
    import pijepa_toolkit as T
    return T, gridspec, mo, np, plt


@app.cell(hide_code=True)
def _rcparams(plt):
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#546E7A",
        "axes.labelcolor":    "#212121",
        "axes.titlesize":     11,
        "axes.titleweight":   "normal",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        "#546E7A",
        "ytick.color":        "#546E7A",
        "grid.color":         "#ECEFF1",
        "grid.linewidth":     0.6,
        "font.family":        "sans-serif",
        "font.size":          10,
        "mathtext.fontset":   "cm",
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.Html("""
    <div style="text-align:center; padding: 0.5rem 0 0.3rem;">
      <h1 style="font-family: Georgia, serif; font-weight: 400; font-size: 3.0rem;
                 margin: 0; line-height: 1.1; color:#0D3349;">
        Closing the Simulation&nbsp;Budget&nbsp;Gap
      </h1>
      <p style="font-family: Georgia, serif; font-size: 1.3rem; color:#455A64;
                margin: 0.4rem 0 0; text-align:center;">
        PI-JEPA: Physics-Informed JEPA for Data-Efficient PDE Surrogates
      </p>
      <p style="font-size: 0.85rem; color:#78909C; margin-top: 0.5rem; text-align:center;">
        <span style="color:#546E7A;">Notebook:</span>
        Brandon&nbsp;Yee &middot; Pairie&nbsp;Koh &middot; Jacob&nbsp;Crainic &middot; Peter&nbsp;Li
        &nbsp;|&nbsp;
        Yee&nbsp;Collins&nbsp;Research&nbsp;Group / Hoover&nbsp;Institution&nbsp;at&nbsp;Stanford
        &nbsp;|&nbsp;
        <a href="https://arxiv.org/abs/2604.01349" style="color:#1565C0;">arXiv:2604.01349</a>
      </p>
    </div>
    """))
    return


@app.cell(hide_code=True)
def _hero_controls(mo):
    hero_seed = mo.ui.slider(0, 19, value=4, show_value=False)
    return (hero_seed,)


@app.cell(hide_code=True)
def _hero_plot(T, gridspec, hero_seed, mo, np, plt):
    _CACHE = T.load_darcy_cache()

    def _draw():
        n    = 32
        seed = int(hero_seed.value)

        if _CACHE is not None:
            K   = _CACHE[f"h_K_{seed}"]
            p   = _CACHE[f"h_p_{seed}"]
            ux  = _CACHE[f"h_ux_{seed}"]
            uy  = _CACHE[f"h_uy_{seed}"]
            ctx = _CACHE[f"h_ctx_{seed}"].astype(bool)
            tgt = _CACHE[f"h_tgt_{seed}"].astype(bool)
        else:
            K  = T.make_permeability_channelized(n, n_layers=3, seed=seed)
            p  = T.solve_darcy_fd(K)
            ux, uy = T.darcy_velocity(K, p)
            ctx, tgt = T.spatiotemporal_block_mask(n, context_frac=0.65, seed=seed)

        speed = np.sqrt(ux**2 + uy**2)

        fig = plt.figure(figsize=(14.5, 4.0), constrained_layout=True)
        gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)
        ext = [0, 1, 0, 1]
        xs  = np.linspace(0, 1, n)

        ax0 = fig.add_subplot(gs[0])
        im0 = ax0.imshow(np.log10(K).T, origin="lower", cmap="RdYlGn",
                          extent=ext, vmin=-2, vmax=2)
        fig.colorbar(im0, ax=ax0, label="log₁₀ K", shrink=0.85)
        ax0.set_title("Permeability  $K$\n(geostat. draw — free)", fontsize=10)
        ax0.set_xlabel("$x$"); ax0.set_ylabel("$y$")

        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.imshow(p.T, origin="lower", cmap="Blues_r",
                          extent=ext, vmin=0, vmax=1)
        fig.colorbar(im1, ax=ax1, label="pressure $p$", shrink=0.85)
        ax1.set_title("Pressure  $p$\n(PDE solve — costly)", fontsize=10)
        ax1.set_xlabel("$x$")

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(speed.T, origin="lower", cmap="hot", extent=ext)
        s = 4
        X2, Y2 = np.meshgrid(xs[::s], xs[::s], indexing="ij")
        ax2.quiver(X2, Y2, ux[::s, ::s].T, uy[::s, ::s].T,
                    scale=4, color="white", alpha=0.75, width=0.006)
        ctx_ma = np.ma.masked_where(~ctx.T, np.ones((n, n)))
        tgt_ma = np.ma.masked_where(~tgt.T, np.ones((n, n)))
        ax2.imshow(ctx_ma, origin="lower", extent=ext,
                    cmap="Blues", alpha=0.45, vmin=0, vmax=1)
        ax2.imshow(tgt_ma, origin="lower", extent=ext,
                    cmap="Reds",  alpha=0.55, vmin=0, vmax=1)
        ax2.set_title("Velocity + PI-JEPA masks\n(blue=context, red=target)", fontsize=10)
        ax2.set_xlabel("$x$")

        ax3 = fig.add_subplot(gs[3])
        nl  = np.array(T.N_LABELS)
        ax3.semilogy(nl, T.DARCY["FNO"],    "^--", color=T.C_FNO,
                      lw=2, ms=6, label="FNO")
        ax3.semilogy(nl, T.DARCY["PI-JEPA"], "o-", color=T.C_PIJEPA,
                      lw=2.5, ms=7, label="PI-JEPA", zorder=5, clip_on=False)
        lo = np.array(T.DARCY["PI-JEPA"]) - np.array(T.DARCY_CI["PI-JEPA"])
        hi = np.array(T.DARCY["PI-JEPA"]) + np.array(T.DARCY_CI["PI-JEPA"])
        ax3.fill_between(nl, np.clip(lo, 1e-4, None), hi,
                          color=T.C_PIJEPA, alpha=0.18)
        ax3.axvline(100, color=T.C_ACCENT, lw=1.6, ls=":", alpha=0.85)
        ax3.text(105, 0.055, "1.8×\nless error\nat $N_\\ell$=100",
                  color=T.C_ACCENT, fontsize=9, fontweight="bold")
        ax3.set_xlabel("Labeled runs  $N_\\ell$")
        ax3.set_ylabel("Rel.  $\\ell_2$  error")
        ax3.set_title("Data efficiency\n(Darcy benchmark)", fontsize=10)
        ax3.legend(fontsize=9, frameon=False)
        ax3.set_xticks(nl[::2])

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _hero_row(hero_seed, mo):
    mo.hstack([
        mo.vstack([
            mo.md("**Permeability seed**"),
            hero_seed,
            mo.md("<span style='font-size:0.78rem;color:#78909C'>0 — 19</span>"),
        ], gap=0.3),
        mo.md(r"""
        Each panel updates instantly because permeability fields are drawn from a geostatistical
        model in milliseconds. The pressure field requires a full FDM solve of $-\nabla\cdot(K\nabla p)=0$,
        which in a real reservoir workflow takes 1–24 hours per run. **PI-JEPA pretrains entirely on
        the cheap $K$ fields; a handful of expensive runs handle fine-tuning.**
        """),
    ], gap=2.5, align="start")
    return


@app.cell(hide_code=True)
def _intro_prose(mo):
    mo.md(r"""
    Three landmarks from the figure above set the scene. First, the spatial structure
    of $K$ — channelized high-permeability streaks threading through a low-permeability
    background — is the primary determinant of how pressure and saturation evolve, yet
    generating that field costs almost nothing. Second, converting a single $K$ field
    into a labeled $(K, p)$ pair requires invoking a numerical PDE solver: hours of
    wall-clock time per run, not milliseconds. Third, even at only 100 labeled runs,
    PI-JEPA already achieves 1.8× lower relative $\ell_2$ error than the Fourier Neural
    Operator on single-phase Darcy flow.

    The rest of this notebook works through why, starting from the data asymmetry that
    makes the whole approach possible.
    """)
    return


@app.cell(hide_code=True)
def _asym_md(mo):
    mo.md(r"""
    ## The Simulation Data Asymmetry

    Every existing neural-operator surrogate — FNO, PINO, DeepONet — treats surrogate
    modeling as supervised reconstruction: given labeled pairs
    $\{(\mathbf{K}^{(i)}, \mathbf{p}^{(i)})\}_{i=1}^{N_\ell}$, minimize pixelwise
    error. That framing ignores a structural feature of reservoir workflows that is
    completely free to exploit.
    """)
    return


@app.cell(hide_code=True)
def _asym_controls(mo):
    cost_nl = mo.ui.slider(10, 500, value=100, step=10,
                            label="Labeled simulations  $N_\\ell$")
    cost_nu = mo.ui.slider(100, 10000, value=1000, step=100,
                            label="Unlabeled K fields  $N_u$")
    return cost_nl, cost_nu


@app.cell(hide_code=True)
def _asym_plot(T, cost_nl, cost_nu, mo, plt):
    def _draw():
        nl = cost_nl.value
        nu = cost_nu.value
        cm = T.cost_model(n_unlabeled=nu, n_labeled=nl)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)

        cats  = ["Unlabeled\n(geostat.)", "Labeled\n(PDE solve)"]
        costs = [cm["unlabeled"]["total_cost_h"], cm["labeled"]["total_cost_h"]]
        cols  = [T.C_PRETRAIN, T.C_FNO]
        bars  = axes[0].bar(cats, costs, color=cols, alpha=0.88, width=0.5,
                              edgecolor="white", linewidth=1.5)
        axes[0].set_yscale("log")
        y_min, y_max = 1e-3, max(costs) * 10
        axes[0].set_ylim(y_min, y_max)
        for bar, h in zip(bars, costs):
            label = f"{h:.4f} h" if h < 0.01 else (f"{h:.2f} h" if h < 10 else f"{h:.0f} h")
            label_y = max(h * 2.0, y_min * 3)
            axes[0].text(bar.get_x() + bar.get_width() / 2, label_y,
                          label, ha="center", va="bottom",
                          fontsize=11, fontweight="bold",
                          color="white" if h < 0.01 else "black")
        axes[0].set_ylabel("Total CPU hours")
        axes[0].set_title(
            f"Data generation cost\n$N_u$={nu:,} unlabeled  vs  $N_\\ell$={nl} labeled",
            fontsize=11)

        unit = [cm["unlabeled"]["unit_cost_s"], cm["labeled"]["unit_cost_s"]]
        axes[1].bar(cats, unit, color=cols, alpha=0.88, width=0.5,
                     edgecolor="white", linewidth=1.5)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("CPU seconds per sample")
        axes[1].set_title(
            f"Per-sample cost ratio: {cm['ratio']:,.0f}×\n"
            f"(one labeled run ≈ one full working day)",
            fontsize=11)
        axes[1].text(1, unit[1] / 4, f"×{cm['ratio']:,.0f}",
                      color="white", ha="center", fontsize=14, fontweight="bold")

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _asym_sliders(cost_nl, cost_nu):
    cost_nl
    cost_nu
    return


@app.cell(hide_code=True)
def _asym_prose(mo):
    mo.md(r"""
    Permeability realizations are drawn from geostatistical models — sequential Gaussian
    simulation, multi-point statistics, training-image methods — in milliseconds, in
    arbitrarily large quantities. These unlabeled fields encode the spatial heterogeneity
    structure of the subsurface, the single most important determinant of flow behavior,
    yet no existing neural-operator framework can pretrain on them.

    PI-JEPA is designed around this asymmetry. Its pretraining phase requires only
    unlabeled $\mathbf{K}$ fields and invokes no PDE solver. The labeled budget is
    spent entirely on fine-tuning, and the architecture is designed so that even 50–100
    labeled runs are sufficient to achieve strong surrogate accuracy.
    """)
    return


@app.cell(hide_code=True)
def _arch_md(mo):
    mo.md(r"""
    ## The PI-JEPA Framework

    PI-JEPA has three learned components: a **context encoder** $f_\theta$ (Fourier-enhanced backbone), an **EMA target encoder** $f_\xi$, and a **predictor bank** $\{g_{\phi_k}\}_{k=1}^K$ with one predictor per physical sub-operator. The total pretraining loss is

    $$\mathcal{L} = \underbrace{\mathcal{L}_{\text{pred}}}_{\text{latent prediction}} + \lambda_p \sum_{k=1}^K \underbrace{\mathcal{L}_{\text{phys}}^{(k)}}_{\text{PDE residual}} + \lambda_r \underbrace{\mathcal{L}_{\text{VICReg}}}_{\text{collapse prevention}}$$

    No labeled trajectories enter the pretraining objective. The predictor bank structure — with one module per sub-operator — is the central architectural contribution, identified as responsible for a 16% performance gain in ablation.
    """)
    return


@app.cell(hide_code=True)
def _arch_html(mo):
    mo.center(mo.Html("""
    <div style="font-family:'Helvetica Neue',Arial,sans-serif; background:#F8FAFC;
                border-radius:12px; padding:22px 28px; max-width:900px;
                margin:0 auto; border:1.5px solid #B0BEC5;">

      <div style="text-align:center; font-size:0.73rem; color:#78909C;
                  letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;">
        Phase 1 — Pretrain on unlabeled <b>K</b> fields (no PDE solves)
      </div>

      <div style="display:flex; align-items:center; gap:9px; justify-content:center; flex-wrap:wrap;">

        <div style="background:#E8F5E9; border:1.5px solid #66BB6A; border-radius:8px;
                    padding:9px 16px; text-align:center; min-width:100px;">
          <div style="font-size:0.78rem; color:#388E3C;">Unlabeled fields</div>
          <div style="font-size:1.2rem; font-weight:bold; color:#1B5E20;">K<sup>(i)</sup></div>
          <div style="font-size:0.68rem; color:#66BB6A;">milliseconds each</div>
        </div>

        <div style="font-size:1.4rem; color:#546E7A;">&#8594;</div>

        <div style="background:#E3F2FD; border:1.5px solid #42A5F5; border-radius:8px;
                    padding:9px 13px; text-align:center; min-width:128px;">
          <div style="font-size:0.78rem; color:#1565C0;">Fourier Encoder</div>
          <div style="font-size:1.1rem; font-weight:bold; color:#0D47A1;">f<sub>&theta;</sub></div>
          <div style="font-size:0.68rem; color:#42A5F5;">6 Fourier + 4 attention layers<br>d=384 patch embeddings</div>
        </div>

        <div style="font-size:1.4rem; color:#546E7A;">&#8594;</div>

        <div style="border:1.5px dashed #FFA726; border-radius:8px;
                    padding:8px 12px; background:#FFF8E1; text-align:center;">
          <div style="font-size:0.72rem; color:#E65100; margin-bottom:5px;">Predictor Bank (K=2)</div>
          <div style="display:flex; gap:6px; justify-content:center;">
            <div style="background:#FFE082; border-radius:5px; padding:4px 10px;
                        font-size:0.84rem; font-weight:bold; color:#E65100;">
              g<sub>&phi;&#8321;</sub><br>
              <span style="font-size:0.63rem; font-weight:normal;">L&#8321;: pressure</span>
            </div>
            <div style="background:#FFCC02; border-radius:5px; padding:4px 10px;
                        font-size:0.84rem; font-weight:bold; color:#E65100;">
              g<sub>&phi;&#8322;</sub><br>
              <span style="font-size:0.63rem; font-weight:normal;">L&#8322;: saturation</span>
            </div>
          </div>
          <div style="font-size:0.63rem; color:#BF360C; margin-top:4px;">Lie&#8211;Trotter aligned</div>
        </div>

        <div style="font-size:1.4rem; color:#546E7A;">&#8594;</div>

        <div style="background:#FCE4EC; border:1.5px solid #EC407A; border-radius:8px;
                    padding:9px 13px; text-align:center; min-width:114px;">
          <div style="font-size:0.78rem; color:#880E4F;">EMA Target</div>
          <div style="font-size:1.1rem; font-weight:bold; color:#880E4F;">f<sub>&xi;</sub></div>
          <div style="font-size:0.68rem; color:#EC407A;">&xi; &larr; 0.996 &xi; + 0.004 &theta;</div>
        </div>

        <div style="font-size:1.3rem; color:#546E7A;">+</div>

        <div style="background:#EDE7F6; border:1.5px solid #7E57C2; border-radius:8px;
                    padding:9px 13px; text-align:center; min-width:114px;">
          <div style="font-size:0.78rem; color:#4527A0;">VICReg</div>
          <div style="font-size:0.70rem; color:#7E57C2;">var + cov regularizer<br>&rarr; prevents collapse<br>+17.7% in ablation</div>
        </div>
      </div>

      <hr style="border:none; border-top:1px solid #CFD8DC; margin:16px 0;">

      <div style="text-align:center; font-size:0.73rem; color:#78909C;
                  letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;">
        Phase 2 &mdash; Fine-tune on N<sub>&ell;</sub> labeled runs
      </div>

      <div style="display:flex; align-items:center; gap:9px; justify-content:center; flex-wrap:wrap;">

        <div style="background:#E8F5E9; border:1.5px solid #66BB6A; border-radius:8px;
                    padding:9px 16px; text-align:center; min-width:100px;">
          <div style="font-size:0.78rem; color:#388E3C;">Labeled pairs</div>
          <div style="font-size:1.0rem; font-weight:bold; color:#1B5E20;">(K, p)<sup>(j)</sup></div>
          <div style="font-size:0.68rem; color:#66BB6A; font-style:italic;">1&ndash;24 h each</div>
        </div>

        <div style="font-size:1.4rem; color:#546E7A;">&#8594;</div>

        <div style="background:#E3F2FD; border:2px solid #1565C0; border-radius:8px;
                    padding:9px 13px; text-align:center; min-width:128px;">
          <div style="font-size:0.78rem; color:#1565C0;">Pretrained f<sub>&theta;</sub></div>
          <div style="font-size:0.70rem; color:#42A5F5;">full fine-tune, LR&times;0.2<br>300 epochs, AdamW</div>
        </div>

        <div style="font-size:1.4rem; color:#546E7A;">&#8594;</div>

        <div style="background:#F3E5F5; border:1.5px solid #AB47BC; border-radius:8px;
                    padding:9px 13px; text-align:center; min-width:108px;">
          <div style="font-size:0.78rem; color:#6A1B9A;">Prediction Head</div>
          <div style="font-size:0.70rem; color:#AB47BC;">unpatchify + 2&times;conv &rarr; p&#770;</div>
        </div>

        <div style="font-size:1.4rem; color:#546E7A;">&#8594;</div>

        <div style="background:#E8EAF6; border:1.5px solid #3F51B5; border-radius:8px;
                    padding:9px 16px; text-align:center; min-width:108px;">
          <div style="font-size:0.78rem; color:#1A237E;">Output</div>
          <div style="font-size:1.1rem; font-weight:bold; color:#1A237E;">p&#770;</div>
          <div style="font-size:0.68rem; color:#3F51B5;">1.8&ndash;2.7&times; better than FNO</div>
        </div>
      </div>
    </div>
    """))
    return


@app.cell(hide_code=True)
def _split_md(mo):
    mo.md(r"""
    ## Operator Splitting: The Physical Inductive Bias

    Classical numerical solvers decompose each time-step via the Lie–Trotter splitting

    $$u^{n+1} \approx \mathcal{L}_K(\Delta t) \circ \mathcal{L}_S(\Delta t)\, u^n,$$

    where $\mathcal{L}_K$ is the **elliptic pressure sub-step** (globally coupled, solved
    implicitly) and $\mathcal{L}_S$ is the **hyperbolic saturation transport sub-step**
    (local, explicit). PI-JEPA mirrors this exactly: predictor $g_{\phi_1}$ advances the
    latent through the pressure sub-step; $g_{\phi_2}$ advances through saturation. Each
    has its own PDE residual as additional supervision.

    Removing operator splitting degrades performance by **+16%** in the ablation study.
    Each predictor need only capture the dynamics of a single physical timescale rather
    than the coupled multi-scale system, which is what makes label-free pretraining tractable.
    """)
    return


@app.cell(hide_code=True)
def _split_controls(mo):
    sp_seed = mo.ui.slider(0, 29, value=7, show_value=False)
    sp_typ  = mo.ui.dropdown(options=["GRF", "channelized"], value="channelized",
                              label="K type")
    return sp_seed, sp_typ


@app.cell(hide_code=True)
def _split_plot(T, gridspec, mo, np, plt, sp_seed, sp_typ):
    _CACHE = T.load_darcy_cache()

    def _draw():
        n    = 32
        seed = int(sp_seed.value)
        kind = sp_typ.value

        prefix = "sg" if kind == "GRF" else "sc"
        if _CACHE is not None and seed < 30:
            K  = _CACHE[f"{prefix}_K_{seed}"]
            p  = _CACHE[f"{prefix}_p_{seed}"]
            ux = _CACHE[f"{prefix}_ux_{seed}"]
            uy = _CACHE[f"{prefix}_uy_{seed}"]
        else:
            K = (T.make_permeability_grf(n, length_scale=0.20, variance=2.0, seed=seed)
                  if kind == "GRF"
                  else T.make_permeability_channelized(n, n_layers=3, seed=seed))
            p = T.solve_darcy_fd(K)
            ux, uy = T.darcy_velocity(K, p)

        xs = np.linspace(0, 1, n)
        XX, YY = np.meshgrid(xs, xs, indexing="ij")
        S0 = np.exp(-((XX - 0.5)**2 + (YY - 0.5)**2) / 0.08**2)
        dx = 1.0 / (n - 1)
        dSdx = np.gradient(S0, dx, axis=0)
        dSdy = np.gradient(S0, dx, axis=1)
        S1   = S0 - 0.05 * (ux * dSdx + uy * dSdy)
        dS   = S1 - S0

        ctx_L1, tgt_L2 = T.operator_split_masks(n, seed=seed)

        fig = plt.figure(figsize=(15.5, 4.3), constrained_layout=True)
        gs2  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.07)
        ext  = [0, 1, 0, 1]
        s    = max(1, n // 10)
        X2, Y2 = np.meshgrid(xs[::s], xs[::s], indexing="ij")

        ax = fig.add_subplot(gs2[0])
        ax.imshow(np.log10(K).T, origin="lower", cmap="RdYlGn",
                   extent=ext, vmin=-2, vmax=2)
        ax.set_title("Permeability  $K$\n(input — free)", fontsize=10)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")

        ax = fig.add_subplot(gs2[1])
        im1 = ax.imshow(p.T, origin="lower", cmap="Blues_r", extent=ext)
        plt.colorbar(im1, ax=ax, shrink=0.82)
        ax.set_title("$\\mathcal{L}_1$: pressure solve\n(elliptic — globally coupled)",
                      fontsize=10, color=T.C_PIJEPA)
        ax.set_xlabel("$x$")

        ax = fig.add_subplot(gs2[2])
        ax.imshow(S0.T, origin="lower", cmap="Oranges", extent=ext, alpha=0.65)
        ax.quiver(X2, Y2, ux[::s, ::s].T, uy[::s, ::s].T,
                   scale=3.5, color=T.C_PIJEPA, alpha=0.85, width=0.007)
        ax.set_title("$\\mathcal{L}_2$: saturation transport\n(hyperbolic — local/explicit)",
                      fontsize=10, color=T.C_FNO)
        ax.set_xlabel("$x$")

        ax = fig.add_subplot(gs2[3])
        ax.imshow(S0.T, origin="lower", cmap="Oranges", extent=ext)
        overlay_ctx = np.ma.masked_where(~ctx_L1.T, np.ones((n, n)))
        ax.imshow(overlay_ctx, origin="lower", extent=ext,
                   cmap="Blues", alpha=0.45)
        ax.set_title("PI-JEPA context → $g_{\\phi_1}$\n(pressure context region)",
                      fontsize=10, color=T.C_PRETRAIN)
        ax.set_xlabel("$x$")

        ax = fig.add_subplot(gs2[4])
        im4 = ax.imshow(dS.T, origin="lower", cmap="RdBu_r",
                          extent=ext, vmin=-0.15, vmax=0.15)
        plt.colorbar(im4, ax=ax, shrink=0.82, label="$\\Delta S$")
        ax.set_title("PI-JEPA target → $g_{\\phi_2}$\n(saturation update $\\Delta S$)",
                      fontsize=10, color=T.C_FNO)
        ax.set_xlabel("$x$")

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _split_widgets(mo, sp_seed, sp_typ):
    mo.hstack([
        mo.vstack([mo.md("**Permeability seed**"), sp_seed], gap=0.3),
        sp_typ,
    ], gap=2.0, align="start")
    return


@app.cell(hide_code=True)
def _de_md(mo):
    mo.md(r"""
    ## Data Efficiency: The Core Result

    Three benchmark PDE systems, each evaluated as a function of the number of labeled
    fine-tuning samples $N_\ell$. Mean ± 95% CI over 5 random seeds. All baselines are
    trained from scratch on the same labeled budget with identical epoch counts.
    """)
    return


@app.cell(hide_code=True)
def _de_controls(T, mo):
    bench_sel = mo.ui.dropdown(
        options=list(T.ALL_BENCHMARKS.keys()),
        value="Darcy (single-phase)",
        label="Benchmark",
    )
    methods_sel = mo.ui.multiselect(
        options=["PI-JEPA", "Scratch", "FNO", "PINO", "DeepONet"],
        value=["PI-JEPA", "Scratch", "FNO", "PINO"],
        label="Methods",
    )
    nl_line = mo.ui.slider(
        steps=T.N_LABELS,
        value=100,
        label="Highlight  $N_\\ell$",
    )
    return bench_sel, methods_sel, nl_line


@app.cell(hide_code=True)
def _de_plot(T, bench_sel, methods_sel, mo, nl_line, np, plt):
    _METHOD_STYLE = {
        "PI-JEPA":  dict(color=T.C_PIJEPA,   lw=3.0, marker="o", ms=8,  ls="-"),
        "Scratch":  dict(color=T.C_SCRATCH,   lw=2.0, marker="s", ms=6,  ls="--"),
        "FNO":      dict(color=T.C_FNO,       lw=2.0, marker="^", ms=6,  ls="--"),
        "PINO":     dict(color=T.C_PINO,      lw=1.8, marker="v", ms=5,  ls=":"),
        "DeepONet": dict(color=T.C_DEEPONET,  lw=1.8, marker="D", ms=5,  ls="-."),
    }

    def _draw():
        bname = bench_sel.value
        data, ci = T.ALL_BENCHMARKS[bname]
        nl    = np.array(T.N_LABELS)
        mth   = methods_sel.value or ["PI-JEPA", "FNO"]
        nl_h  = nl_line.value

        fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(12.5, 4.5),
                                        constrained_layout=True)

        for m in mth:
            kw = _METHOD_STYLE[m]
            y  = np.array(data[m])
            ye = np.array(ci[m])
            ax.semilogy(nl, y, label=m, clip_on=False, **kw)
            ax.fill_between(nl, np.clip(y - ye, 1e-4, None), y + ye,
                              color=kw["color"], alpha=0.14)

        ax.axvline(nl_h, color=T.C_ACCENT, lw=1.8, ls=":", alpha=0.9)

        if "PI-JEPA" in mth and "FNO" in mth:
            idx_h = T.N_LABELS.index(nl_h) if nl_h in T.N_LABELS else 0
            v_pj  = data["PI-JEPA"][idx_h]
            v_fno = data["FNO"][idx_h]
            if v_fno > v_pj:
                factor = v_fno / v_pj
                ax.text(nl_h * 1.06, v_pj * 1.5,
                         f"{factor:.1f}×\nless error",
                         color=T.C_ACCENT, fontsize=9.5, fontweight="bold",
                         va="bottom")

        ax.set_xlabel("Labeled simulations  $N_\\ell$")
        ax.set_ylabel("Relative $\\ell_2$ error  (↓ better)")
        ax.set_title(f"Data Efficiency — {bname}", fontsize=12)
        ax.legend(fontsize=9, frameon=False)
        ax.set_xticks(nl)
        ax.set_xticklabels([str(x) for x in nl], fontsize=8)

        if "PI-JEPA" in mth and "FNO" in mth:
            y_pj  = np.array(data["PI-JEPA"])
            y_fno = np.array(data["FNO"])
            ratio = y_fno / (y_pj + 1e-10)
            ax_r.plot(nl, ratio, "o-", color=T.C_PIJEPA, lw=2.5, ms=7, clip_on=False)
            ax_r.axhline(1.0, color="#90A4AE", lw=1.3, ls="--")
            ax_r.fill_between(nl, 1, ratio, where=(ratio > 1),
                               color=T.C_PIJEPA, alpha=0.18, label="PI-JEPA wins")
            ax_r.fill_between(nl, 1, ratio, where=(ratio < 1),
                               color=T.C_FNO,    alpha=0.18, label="FNO wins")
            ax_r.axvline(nl_h, color=T.C_ACCENT, lw=1.8, ls=":", alpha=0.9)
            ax_r.set_xlabel("Labeled simulations  $N_\\ell$")
            ax_r.set_ylabel("FNO error / PI-JEPA error  (>1 = PI-JEPA wins)")
            ax_r.set_title("Improvement ratio (PI-JEPA vs FNO)", fontsize=12)
            ax_r.set_xticks(nl)
            ax_r.set_xticklabels([str(x) for x in nl], fontsize=8)
            ax_r.legend(fontsize=9, frameon=False)
        else:
            ax_r.set_visible(False)

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _de_widgets(bench_sel, methods_sel, mo, nl_line):
    mo.hstack([bench_sel, methods_sel, nl_line], gap=1.5)
    return


@app.cell(hide_code=True)
def _de_findings(mo):
    mo.md(r"""
    Three findings stand out across the benchmarks.

    On **Darcy**, PI-JEPA achieves 1.8× lower error than FNO at $N_\ell = 100$.
    PINO performs nearly identically to FNO across all $N_\ell$, demonstrating that
    adding PDE residuals to a supervised neural operator provides negligible benefit
    when labels are scarce. This motivates using physics constraints during unsupervised
    *pretraining* rather than supervised fine-tuning.

    On **two-phase CO₂-water**, FNO and PINO plateau above a relative $\ell_2$ error
    of 1.15 even at $N_\ell = 500$. They cannot capture the sharp saturation fronts
    characteristic of immiscible displacement regardless of training-set size. PI-JEPA
    achieves 2.7× lower error at $N_\ell = 100$.

    At $N_\ell \geq 250$ on Darcy, FNO and PINO surpass PI-JEPA. Single-phase Darcy is
    a single elliptic PDE, and spectral convolutions are specifically designed for this
    problem class. PI-JEPA's advantage is data efficiency in the label-scarce regime, not
    asymptotic accuracy.
    """)
    return


@app.cell(hide_code=True)
def _abl_md(mo):
    mo.md(r"""
    ## Ablation: Which Components Matter?

    The paper systematically removes each PI-JEPA component on the Darcy benchmark at
    $N_\ell = 100$ (5 seeds). Positive $\Delta$ means removing a component *increases*
    error — the component is **essential**. A negative $\Delta$ means removing it
    *decreases* error — an **honest negative result**.
    """)
    return


@app.cell(hide_code=True)
def _abl_plot(T, mo, np, plt):
    def _draw():
        items  = list(T.ABLATION.items())
        names  = [k for k, _ in items]
        means  = np.array([v[0] for _, v in items])
        cis    = np.array([v[1] for _, v in items])
        deltas = means - means[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.0),
                                        constrained_layout=True)

        cols = [T.C_PIJEPA] + [
            T.C_FNO if d > 0 else T.C_PRETRAIN for d in deltas[1:]
        ]
        ax1.barh(range(len(names)), means, xerr=cis, color=cols, alpha=0.85,
                  edgecolor="white", lw=1.2, height=0.58,
                  error_kw=dict(ecolor="#546E7A", capsize=4, capthick=1.5))
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9.5)
        ax1.axvline(means[0], color=T.C_PIJEPA, lw=2, ls="--", alpha=0.7)
        ax1.set_xlabel("Relative $\\ell_2$ error  at  $N_\\ell=100$")
        ax1.set_title("Ablation: absolute error", fontsize=12)
        ax1.set_xlim(0.15, 0.245)

        col_r  = [T.C_FNO if d > 0.001 else (T.C_PRETRAIN if d < -0.003 else "#90A4AE")
                   for d in deltas]
        col_r[0] = "#90A4AE"
        ax2.barh(range(len(names)), deltas * 100, color=col_r, alpha=0.88,
                  edgecolor="white", lw=1.2, height=0.58)
        ax2.axvline(0, color="#546E7A", lw=1.5)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9.5)
        ax2.set_xlabel("Δ error vs full PI-JEPA  (%)")
        ax2.set_title("Δ from full model\n(red = essential; teal = honest negative)",
                       fontsize=12)

        phr = names.index("w/o physics residual")
        ax2.text(0.5, phr,
                  "removing it helps (−8.1%) →",
                  fontsize=8, color=T.C_PRETRAIN, va="center", ha="left")

        ax2.text(deltas[2] * 100 + 0.3, 2,
                  f"+{deltas[2]*100:.1f}%  essential",
                  fontsize=8.5, color=T.C_FNO, va="center")
        ax2.text(deltas[3] * 100 + 0.3, 3,
                  f"+{deltas[3]*100:.1f}%  essential",
                  fontsize=8.5, color=T.C_FNO, va="center")

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _abl_prose(mo):
    mo.md(r"""
    **Operator splitting (+16%)** and **VICReg (+17.7%)** are the essential components.
    Together they account for nearly all of PI-JEPA's advantage over a monolithic baseline.
    The operator splitting result confirms that the structured predictor bank aligned to the
    physical decomposition is the primary architectural contribution. The VICReg result
    demonstrates that collapse prevention is critical in the physics-informed latent
    prediction setting, where the PDE residual loss can inadvertently encourage low-rank
    representations if unchecked.

    The **physics residual is neutral** — removing it slightly improves performance (−8.1%),
    an honest finding that contrasts with the theoretical motivation. The paper hypothesizes
    that finite-difference approximation of the residual on a coarse 32×32 collocation grid
    introduces discretization artifacts. PINO shows the same behavior: physics-informed
    regularization at training time provides negligible benefit when labels are scarce.
    """)
    return


@app.cell(hide_code=True)
def _sc_md(mo):
    mo.md(r"""
    ## Why It Works: Sample Complexity Theory

    Proposition 1 of the paper formalizes the data-efficiency advantage in a linear
    latent dynamics model. Let $A = A_K \cdots A_1 \in \mathbb{R}^{n \times n}$ be the
    true operator and $f_\theta : \mathbb{R}^n \to \mathbb{R}^d$ the pretrained encoder:

    | Method | Free parameters | Sample complexity |
    |---|---|---|
    | **Supervised baseline** | $n^2$ | $\Omega(n^2 / \varepsilon^2)$ |
    | **PI-JEPA fine-tuning** | $d^2 K$ | $O(d^2 K / \varepsilon^2)$ |

    For the Darcy setting ($n = 64$, so $n^2 = 4{,}096$; $d = 384$; $K = 2$) the
    ratio is $4{,}096 / (384^2 \times 2) \approx 0.014$.  Pretraining compresses the
    $n^2$-dimensional estimation problem into $K$ sub-problems of dimension $d^2$,
    each informed by $N_u \gg N_\ell$ unlabeled samples. Adjust the parameters below
    to explore the theoretical advantage across settings.
    """)
    return


@app.cell(hide_code=True)
def _sc_controls(mo):
    sc_n = mo.ui.slider(steps=[16, 32, 48, 64, 96, 128], value=64,
                         label="Grid size  $n$")
    sc_d = mo.ui.slider(steps=[32, 64, 128, 256, 384, 512], value=384,
                         label="Latent dim  $d$")
    sc_K = mo.ui.slider(1, 4, value=2, step=1,
                         label="Sub-operators  $K$")
    return sc_K, sc_d, sc_n


@app.cell(hide_code=True)
def _sc_plot(T, mo, np, plt, sc_K, sc_d, sc_n):
    def _draw():
        n  = int(sc_n.value)
        d  = int(sc_d.value)
        K_ = int(sc_K.value)
        stats = T.sample_complexity_advantage(n, d, K_)
        n_grids, d_latents, ratios = T.sample_complexity_surface(K=K_)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5),
                                        constrained_layout=True)

        im = ax1.pcolormesh(d_latents, n_grids, np.log10(ratios + 1e-3),
                             cmap="RdYlGn_r", shading="gouraud")
        fig.colorbar(im, ax=ax1,
                      label="log₁₀(n²/(d²K))  — lower = PI-JEPA needs far fewer labels")
        ax1.plot(d, n, "w*", ms=18, zorder=6)
        ax1.plot(d, n, "k*", ms=8,  zorder=7)
        ax1.text(d + 8, n + 2,
                  f"Current:\n{stats['ratio']:.3f}× ratio",
                  fontsize=9, color="#212121")
        ax1.set_xlabel("Latent dim  $d$")
        ax1.set_ylabel("Grid size  $n$")
        ax1.set_title(f"Sample complexity ratio  $n^2/(d^2 K)$,   $K={K_}$",
                       fontsize=11)

        ax2.axis("off")
        rows = [
            ("Supervised baseline needs", f"$n^2 = {stats['n_params_supervised']:,}$ parameters",
             T.C_FNO),
            ("PI-JEPA fine-tuning needs",
             f"$d^2 K = {stats['n_params_pijepa']:,}$ parameters",
             T.C_PIJEPA),
            ("Advantage",
             f"{stats['ratio']:.3f}× ratio — PI-JEPA needs far fewer labels",
             "#00695C"),
            ("Intuition",
             f"Pretraining turns one $n^2$-dim\nproblem into $K = {K_}$ problems\nof dimension $d^2$",
             "#455A64"),
        ]
        y = 0.88
        for label, val, col in rows:
            ax2.text(0.04, y, label, fontsize=10, color="#546E7A",
                      transform=ax2.transAxes, va="top")
            ax2.text(0.04, y - 0.10, val, fontsize=12.5, color=col,
                      fontweight="bold", transform=ax2.transAxes, va="top")
            y -= 0.24
        ax2.set_title("Current configuration", fontsize=12)

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _sc_widgets(mo, sc_K, sc_d, sc_n):
    mo.hstack([sc_n, sc_d, sc_K], gap=2.0)
    return


@app.cell(hide_code=True)
def _transfer_md(mo):
    mo.md(r"""
    ## Cross-Domain Transfer

    A striking result from Table 4: encoders pretrained on Darcy flow transfer
    remarkably well to the two-phase CO₂-water benchmark, even though pressure
    fields and saturation fronts have qualitatively different spatial structure.
    At $N_\ell = 100$, the Darcy-pretrained encoder (0.410) slightly *outperforms*
    domain-matched pretraining (0.425) — suggesting that the spatial heterogeneity
    structure learned from permeability fields is broadly useful for pressure-dominated
    systems. Domain-matched pretraining only provides additional benefit at $N_\ell \geq 250$.

    On ADR (reaction-dominated dynamics), the domain gap is larger and domain-matched
    pretraining wins more clearly at both label counts tested.
    """)
    return


@app.cell(hide_code=True)
def _transfer_plot(T, mo, plt):
    def _draw():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                        constrained_layout=True)
        tcols = {"Domain-matched":   T.C_PIJEPA,
                 "Darcy-pretrained": T.C_ACCENT,
                 "Scratch":          T.C_SCRATCH}
        tls   = {"Domain-matched": "-", "Darcy-pretrained": "--", "Scratch": ":"}

        for m, vals in T.TRANSFER_2PHASE.items():
            ax1.plot(T.TRANSFER_2PHASE_NL, vals, marker="o", ms=7, lw=2.2,
                      color=tcols[m], ls=tls[m], label=m, clip_on=False)
        ax1.set_title("Two-phase: Darcy→CO₂-water transfer", fontsize=11)
        ax1.set_xlabel("$N_\\ell$"); ax1.set_ylabel("Rel. $\\ell_2$ error")
        ax1.set_xticks(T.TRANSFER_2PHASE_NL)
        ax1.legend(fontsize=9, frameon=False)
        ax1.set_ylim(0.15, 0.50)
        ax1.annotate(
            "Darcy-pre beats\ndomain-matched\nat $N_\\ell$=100!",
            xy=(100, 0.410), xytext=(175, 0.460),
            arrowprops=dict(arrowstyle="->", color=T.C_ACCENT),
            color=T.C_ACCENT, fontsize=8.5)

        for m, vals in T.TRANSFER_ADR.items():
            ax2.plot(T.TRANSFER_ADR_NL, vals, marker="s", ms=7, lw=2.2,
                      color=tcols[m], ls=tls[m], label=m, clip_on=False)
        ax2.set_title("ADR: Darcy→advection-diffusion-reaction", fontsize=11)
        ax2.set_xlabel("$N_\\ell$"); ax2.set_ylabel("Rel. $\\ell_2$ error")
        ax2.set_xticks(T.TRANSFER_ADR_NL)
        ax2.legend(fontsize=9, frameon=False)
        ax2.set_ylim(0.055, 0.105)

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _limits_md(mo):
    mo.md(r"""
    ## Honest Limits

    The paper reports three failure modes transparently. They are worth stating precisely
    because they redirect future work toward the questions that actually matter.
    """)
    return


@app.cell(hide_code=True)
def _limits_plot(T, mo, np, plt):
    def _draw():
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), constrained_layout=True)
        nl = np.array(T.N_LABELS)

        # Panel 1 — physics residual is neutral
        ax = axes[0]
        names4 = ["Full PI-JEPA", "w/o physics residual",
                   "w/o operator splitting", "w/o VICReg"]
        vals4  = [T.ABLATION[n][0] for n in names4]
        cis4   = [T.ABLATION[n][1] for n in names4]
        cols4  = [T.C_PIJEPA, T.C_PRETRAIN, T.C_FNO, T.C_FNO]
        ax.barh(range(4), vals4, xerr=cis4, color=cols4, alpha=0.82,
                 edgecolor="white", lw=1.2, height=0.55,
                 error_kw=dict(ecolor="#546E7A", capsize=4))
        ax.set_yticks(range(4))
        ax.set_yticklabels([n.replace(" w/o", "\nw/o") for n in names4], fontsize=8.5)
        ax.set_xlabel("Rel. $\\ell_2$ error at $N_\\ell=100$")
        ax.set_title("1. Physics residual is neutral:\nremoving it helps by 8.1%", fontsize=11)
        ax.axvline(vals4[0], color=T.C_PIJEPA, lw=2, ls="--", alpha=0.7)
        ax.text(vals4[0] + 0.001, 1, "removing it helps →",
                 fontsize=8, color=T.C_PRETRAIN, va="center", ha="left")

        # Panel 2 — ADR crossover at Nℓ=500
        ax = axes[1]
        ax.semilogy(nl, T.ADR["PI-JEPA"], "o-",  color=T.C_PIJEPA, lw=2.5, ms=7,
                     label="PI-JEPA", clip_on=False)
        ax.semilogy(nl, T.ADR["Scratch"], "s--", color=T.C_SCRATCH, lw=2.0, ms=6,
                     label="Scratch", clip_on=False)
        ax.semilogy(nl, T.ADR["FNO"],     "^:",  color=T.C_FNO,    lw=1.8, ms=5,
                     label="FNO", clip_on=False)
        ax.fill_between(nl[-2:], T.ADR["PI-JEPA"][-2:], T.ADR["Scratch"][-2:],
                         where=np.array(T.ADR["Scratch"][-2:]) < np.array(T.ADR["PI-JEPA"][-2:]),
                         color=T.C_SCRATCH, alpha=0.25)
        ax.set_xlabel("$N_\\ell$"); ax.set_ylabel("Rel. $\\ell_2$ error")
        ax.set_title("2. ADR at $N_\\ell=500$:\nscratch (0.024) beats PI-JEPA (0.065)", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.annotate("Scratch wins!", xy=(500, T.ADR["Scratch"][-1]),
                     xytext=(350, 0.033),
                     arrowprops=dict(arrowstyle="->", color=T.C_SCRATCH),
                     color=T.C_SCRATCH, fontsize=9)

        # Panel 3 — Darcy crossover at Nℓ=250
        ax = axes[2]
        ax.semilogy(nl, T.DARCY["PI-JEPA"], "o-",  color=T.C_PIJEPA, lw=2.5, ms=7,
                     label="PI-JEPA", clip_on=False)
        ax.semilogy(nl, T.DARCY["FNO"],     "^--", color=T.C_FNO,    lw=2.0, ms=6,
                     label="FNO", clip_on=False)
        d_pj  = np.array(T.DARCY["PI-JEPA"])
        d_fno = np.array(T.DARCY["FNO"])
        ax.fill_between(nl[:4], d_pj[:4], d_fno[:4],
                         where=d_pj[:4] < d_fno[:4],
                         color=T.C_PIJEPA, alpha=0.20, label="PI-JEPA wins")
        ax.fill_between(nl[-3:], d_pj[-3:], d_fno[-3:],
                         where=d_fno[-3:] < d_pj[-3:],
                         color=T.C_FNO, alpha=0.20, label="FNO wins")
        ax.axvline(250, color="#78909C", lw=1.5, ls=":")
        ax.text(260, 0.07, "FNO takes over", fontsize=8.5, color=T.C_FNO)
        ax.set_xlabel("$N_\\ell$"); ax.set_ylabel("Rel. $\\ell_2$ error")
        ax.set_title("3. Darcy: FNO wins at $N_\\ell \\geq 250$\n(spectral bias matches problem)",
                      fontsize=11)
        ax.legend(fontsize=9, frameon=False)

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _limits_prose(mo):
    mo.md(r"""
    The three failure modes form a coherent picture. The physics residual requires stronger
    implementation than finite differences on a coarse grid — spectral residuals or
    conservation-law-based losses represent open future work. The ADR crossover at high
    $N_\ell$ reflects a mismatch between pretraining representations (pressure-heterogeneity
    dominated) and the downstream task (reaction-dominated concentration dynamics); a
    domain-matched pretrained encoder recovers most of the gap. The Darcy crossover at
    $N_\ell = 250$ is expected: single-phase Darcy is structurally well-matched to FNO's
    spectral inductive bias, and PI-JEPA's advantage is strictly in the data-scarce regime
    that characterizes real subsurface workflows.
    """)
    return


@app.cell(hide_code=True)
def _sigreg_md(mo):
    mo.md(r"""
    ## Collapse Prevention: VICReg and SIGReg

    PI-JEPA uses **VICReg** (variance-invariance-covariance regularization) to prevent
    representational collapse during the self-supervised pretraining phase. Without it,
    the PDE residual loss can drive the encoder into low-rank representations that
    trivially satisfy the prediction objective. The ablation quantifies this at +17.7%.

    The **LeJEPA** framework on which PI-JEPA builds introduced **SIGReg** — Sketched
    Isotropic Gaussian Regularization (Balestriero & LeCun, 2025). SIGReg projects
    embeddings onto $M$ random unit directions, then applies the Epps-Pulley
    characteristic-function test to match each projection to a standard normal.
    It is theoretically grounded (provably prevents collapse) and computationally
    efficient ($\mathcal{O}(NM)$ time and memory). The interactive plot below
    demonstrates how SIGReg distinguishes isotropic from anisotropic embeddings.
    """)
    return


@app.cell(hide_code=True)
def _sigreg_controls(mo):
    sr_d   = mo.ui.slider(steps=[16, 32, 64, 128, 256], value=64,
                           label="Embedding dim  $d$")
    sr_M   = mo.ui.slider(steps=[8, 16, 32, 64, 128, 256], value=64,
                           label="Projection slices  $M$")
    sr_col = mo.ui.slider(0, 100, value=50, step=5,
                           label="% dimensions collapsed")
    return sr_M, sr_col, sr_d


@app.cell(hide_code=True)
def _sigreg_plot(T, mo, np, plt, sr_M, sr_col, sr_d):
    def _draw():
        d     = int(sr_d.value)
        M     = int(sr_M.value)
        pct   = sr_col.value / 100.0
        n_col = int(pct * d)
        N     = 512
        rng   = np.random.default_rng(42)

        Z_iso   = rng.standard_normal((N, d))
        Z_aniso = Z_iso.copy()
        Z_aniso[:, :n_col] *= 0.005

        loss_iso   = T.sigreg_epps_pulley(Z_iso,   n_slices=M)
        loss_aniso = T.sigreg_epps_pulley(Z_aniso, n_slices=M)
        _, _, eigs_iso   = T.vicreg_diagnostic(Z_iso)
        _, _, eigs_aniso = T.vicreg_diagnostic(Z_aniso)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

        ax = axes[0]
        ax.semilogy(np.arange(d), eigs_iso,   lw=2, color=T.C_PRETRAIN,
                     label="Isotropic (good)")
        ax.semilogy(np.arange(d), eigs_aniso, lw=2, color=T.C_FNO,
                     label=f"Anisotropic ({n_col}/{d} collapsed)")
        ax.axhline(0.05, color=T.C_ACCENT, lw=1.5, ls="--", label="Collapse threshold")
        ax.axvspan(0, max(n_col, 1), alpha=0.10, color=T.C_FNO)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Covariance eigenspectrum\n({n_col}/{d} dims collapsed)", fontsize=11)
        ax.legend(fontsize=8, frameon=False)

        ax = axes[1]
        bars = ax.bar(["Isotropic\n(good)", f"Anisotropic\n({int(pct*100)}% collapsed)"],
                       [loss_iso, loss_aniso],
                       color=[T.C_PRETRAIN, T.C_FNO], alpha=0.85,
                       edgecolor="white", linewidth=1.5, width=0.5)
        for bar, v in zip(bars, [loss_iso, loss_aniso]):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.12,
                     f"{v:.1f}", ha="center", va="bottom",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("SIGReg loss  (↓ = better Gaussian)")
        ax.set_title(f"SIGReg detects collapse\n($M={M}$ projection slices)", fontsize=11)
        ax.set_yscale("log")

        ax = axes[2]
        M_vals   = [4, 8, 16, 32, 64, 128, 256]
        losses_i = [T.sigreg_epps_pulley(Z_iso,   n_slices=m) for m in M_vals]
        losses_a = [T.sigreg_epps_pulley(Z_aniso, n_slices=m) for m in M_vals]
        ax.semilogy(M_vals, losses_i, "o-", color=T.C_PRETRAIN, lw=2, ms=6,
                     label="Isotropic")
        ax.semilogy(M_vals, losses_a, "s-", color=T.C_FNO,      lw=2, ms=6,
                     label=f"Anisotropic ({n_col}/{d} collapsed)")
        ax.axvline(M, color=T.C_ACCENT, lw=1.5, ls=":", label=f"Current $M={M}$")
        ax.set_xlabel("Projection slices  $M$")
        ax.set_ylabel("SIGReg loss")
        ax.set_title("Sensitivity to $M$\n(linear complexity in $M$)", fontsize=11)
        ax.legend(fontsize=8.5, frameon=False)

        return fig

    mo.center(_draw())
    return


@app.cell(hide_code=True)
def _sigreg_widgets(mo, sr_M, sr_col, sr_d):
    mo.hstack([sr_d, sr_M, sr_col], gap=2.0)
    return


@app.cell(hide_code=True)
def _conclusion(mo):
    mo.md(r"""
    ## In Five Lines

    1. **The data asymmetry.** Permeability fields are free to generate in arbitrary
       quantities; labeled PDE trajectories cost hours per run. No existing neural
       operator exploits this gap.

    2. **PI-JEPA closes it.** Pretrain a Fourier-enhanced backbone on unlabeled
       $\mathbf{K}$ fields using masked latent prediction (no PDE solves), then
       fine-tune on as few as 50–100 labeled runs.

    3. **Operator splitting is the key architectural choice** (+16% from ablation).
       Aligning one predictor per sub-operator to the Lie–Trotter decomposition lets
       each module specialize to a single physical timescale.

    4. **VICReg is essential** (+17.7%); the physics residual is not (−8.1%,
       honest negative on coarse collocation grids). Architecture beats physics
       supervision on the benchmarks tested.

    5. **PI-JEPA wins in the data-scarce regime** — 1.8× on Darcy, 2.7× on two-phase
       CO₂-water, 1.9× on ADR at $N_\ell = 100$ — and gracefully yields to FNO/PINO
       at $N_\ell \geq 250$ on elliptic problems where spectral inductive bias dominates.

    ---

    *Paper:* [arXiv:2604.01349](https://arxiv.org/abs/2604.01349) ·
    Yee & Koh, Yee Collins Research Group / Hoover Institution at Stanford, 2026.

    *LeJEPA backbone:* [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) ·
    Balestriero & LeCun, 2025.

    ---
    """)
    return


if __name__ == "__main__":
    app.run()