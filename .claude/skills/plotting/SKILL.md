---
name: plotting
description: >
  Use this skill whenever writing or modifying plotting or figure-generation
  code for the EE597 trellis codes project. Trigger on any request like
  "plot BLER curves", "generate figures", "make plots for the paper",
  "visualize results", "chart", "scatter", or "annotate a point".
  Always read this skill before writing any plotting code — it enforces
  IEEE paper-quality output standards, colorblind-safe palettes, and
  consistent method styling across all figures.
---

# Plotting Skill

## Owned Files

| File | Purpose |
|------|---------|
| `plot_utils.py` | All matplotlib configuration and figure helpers |

---

## 0. Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
```

---

## 1. Style Initialization — Call Once Per Script

Use a **context manager** (never mutate global rcParams directly) so styles
don't bleed across unrelated plots:

```python
sns.set_theme(style="ticks", font_scale=1.0)  # clean base before IEEE overrides

IEEE_RC = {
    'font.family':           'serif',
    'font.serif':            ['Times New Roman'],
    'font.size':             10,
    'axes.labelsize':        11,
    'axes.titlesize':        11,
    'xtick.labelsize':       9,
    'ytick.labelsize':       9,
    'legend.fontsize':       8.5,
    'legend.framealpha':     0.85,    # semi-transparent legend box
    'legend.edgecolor':      '0.8',   # light gray frame
    'legend.fancybox':       False,   # square corners = more formal
    'figure.dpi':            150,     # sharp interactive window
    'savefig.dpi':           300,     # publication quality on save
    'savefig.bbox':          'tight', # never clip edge annotations
    'axes.spines.top':       False,   # cleaner IEEE look
    'axes.spines.right':     False,
    'grid.linewidth':        0.5,
    'grid.alpha':            0.7,
}

# Wrap every figure block in rc_context — styles are scoped, no global leakage
with plt.rc_context(IEEE_RC):
    fig, ax = plt.subplots(...)
    # ... all plot calls ...
    plt.savefig('output.png')
    plt.show()
```

---

## 2. Colorblind-Safe Palette (Paul Tol Bright)

Replace ad-hoc color names with the **Paul Tol bright** palette — safe for
colorblindness, legible in greyscale print, and visually distinct:

```python
TOL = {
    'blue':   '#4477AA',
    'red':    '#EE6677',
    'green':  '#228833',
    'yellow': '#CCBB44',
    'cyan':   '#66CCEE',
    'purple': '#AA3377',
    'grey':   '#BBBBBB',
}
```

Use in place of `'steelblue'`, `'darkgreen'`, `'orange'`, etc.
For line series, cycle in order: `blue -> red -> green -> yellow -> cyan -> purple`

---

## 3. Project Method Style Map

Use consistently across ALL project figures for method identification:

```python
METHOD_STYLE = {
    'B1_mismatched_viterbi':  {'color': TOL['red'],    'marker': 'o',  'ls': '--',  'label': 'Mismatch Viterbi (B1)'},
    'B2_oracle_viterbi':      {'color': TOL['green'],  'marker': 's',  'ls': '--',  'label': 'Oracle Viterbi (B2)'},
    'B5_interference_cancel': {'color': TOL['yellow'], 'marker': '^',  'ls': '--',  'label': 'IC + Viterbi (B5)'},
    'N1_gru_e2e':             {'color': TOL['blue'],   'marker': 'D',  'ls': '-',   'label': 'GRU End-to-End (N1)'},
    'N2_neural_bm':           {'color': TOL['purple'], 'marker': 'v',  'ls': '-',   'label': 'Neural BM (N2)'},
    'S1_searched_gru':        {'color': TOL['cyan'],   'marker': '*',  'ls': '-',   'label': 'Searched + GRU (S1)', 'lw': 2.5},
    'B3_random_trellis':      {'color': TOL['grey'],   'marker': 'x',  'ls': ':',   'label': 'Random Trellis (B3)'},
}
```

---

## 4. Figure & Axes Sizing

| Layout           | `figsize`  | Notes                 |
|------------------|------------|-----------------------|
| Single plot      | `(7, 5.5)` | One IEEE column       |
| Two side-by-side | `(12, 5)`  | Wide dual-panel       |
| Three panels     | `(15, 5)`  | Extend proportionally |

Always close with `fig.tight_layout()`.

---

## 5. Standard BLER Curve — Use for All BLER vs SNR Plots

```python
def plot_bler_vs_snr(
    results: dict[str, list[dict]],   # method_name -> list of estimate_bler outputs
    inr_db: float,
    save_path: Path,
    title: str = "",
    show_ci: bool = True,
) -> None:
    sns.set_theme(style="ticks", font_scale=1.0)

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(figsize=(7, 5.5))

        for method_name, pts in results.items():
            style = METHOD_STYLE.get(method_name, {})
            snr = [p['snr_db'] for p in pts]
            bler = [p['bler'] for p in pts]
            ci = [p['ci_95'] for p in pts]

            ax.semilogy(snr, bler,
                        color=style.get('color', 'black'),
                        marker=style.get('marker', 'o'),
                        ls=style.get('ls', '-'),
                        lw=style.get('lw', 1.5),
                        label=style.get('label', method_name))

            if show_ci:
                bler_arr = np.array(bler)
                ci_arr = np.array(ci)
                ax.fill_between(snr,
                                np.clip(bler_arr - ci_arr, 1e-6, 1),
                                np.clip(bler_arr + ci_arr, 1e-6, 1),
                                alpha=0.15,
                                color=style.get('color', 'black'))

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Block Error Rate (BLER)')
        ax.set_ylim([1e-4, 1.0])
        ax.set_xlim([0, 10])
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_title(f'{title} (INR = {inr_db:.0f} dB)')
        ax.grid(True)

        # Save both PDF (for paper) and PNG (for quick viewing)
        fig.tight_layout()
        fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.png'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path.with_suffix('.pdf')}")
```

---

## 6. INR Sweep Plot — For Robustness Analysis

```python
def plot_bler_vs_inr(
    results: dict[str, list[dict]],   # method_name -> list at fixed SNR, varying INR
    snr_db: float,
    save_path: Path,
) -> None:
    """Shows graceful degradation as interference strengthens."""
    ...
    ax.set_xlabel('INR (dB)')
    ax.set_ylabel('Block Error Rate (BLER)')
    ax.set_title(f'Robustness to interference (SNR = {snr_db:.0f} dB)')
```

---

## 7. Scatter Plots

**Simple scatter:**
```python
ax.scatter(x, y, s=8, color=TOL['blue'], label='Feasible allocations', zorder=2)
```

**Colormap scatter with properly sized colorbar**
(use `make_axes_locatable` so colorbar height matches axes exactly):

```python
sc = ax.scatter(P1, P2, c=values, cmap='viridis', s=10, zorder=2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = fig.colorbar(sc, cax=cax)
cb.set_label(r'$R_1 + R_2$  (bits/s/Hz)')
cb.ax.tick_params(labelsize=8)
```

Recommended cmaps:
- `'viridis'`  — sequential, perceptually uniform, print-safe
- `'RdBu_r'`  — diverging (differences/residuals), colorblind-safe
- `'cividis'` — fully colorblind-safe sequential alternative

---

## 8. Special Marker Conventions

| Point type           | Marker | Size  | Color            |
|----------------------|--------|-------|------------------|
| Optimal / best       | `'*'`  | 16-18 | `TOL['red']`     |
| Fairness / secondary | `'D'`  | 12-13 | `TOL['green']`   |
| Start / initial      | `'s'`  | 10    | `TOL['yellow']`  |
| Trajectory points    | `'o'`  | 3     | `'black'`        |

Always assign `zorder` so markers appear above scatter clouds (use 3-5).

---

## 9. Line Plots & Reference Lines

```python
# Data line
ax.plot(x, y, '-', color=TOL['blue'], linewidth=1.5, label='...')

# Algorithm trajectory
ax.plot(traj_x, traj_y, 'k-o', markersize=3, linewidth=0.8,
        label='Algorithm trajectory', zorder=4)

# Reference line
ax.axvline(x=0, color=TOL['grey'], linestyle='--', linewidth=1,
           label=r'Equal rates ($R_1 = R_2$)', zorder=1)
```

---

## 10. Annotations with Arrows

```python
ax.annotate(
    f'Max $R$ = {val:.2f}\n$P_1$={p1:.2f}, $P_2$={p2:.2f}',
    xy=(x_point, y_point),
    xytext=(x_point + 1.5, y_point + 1.5),
    fontsize=8.5,
    color=TOL['red'],
    fontweight='bold',
    arrowprops=dict(arrowstyle='->', color=TOL['red'], lw=1.2)
)
```

---

## 11. Axis Labels, Titles, Legends, Grid

```python
# Always raw strings for LaTeX math
ax.set_xlabel(r'Rate Difference  $R_2 - R_1$  (bits/s/Hz)')
ax.set_ylabel(r'Sum Rate  $R_1 + R_2$  (bits/s/Hz)')

ax.legend(loc='lower left')   # framealpha/edgecolor handled by IEEE_RC
ax.grid(True)
sns.despine(ax=ax)             # reinforces spine removal if needed
```

---

## 12. Shaded Feasible Regions

```python
p_grid = np.linspace(0, p_max, 300)
X, Y = np.meshgrid(p_grid, p_grid)
feasible = (X >= threshold1) & (Y >= threshold2)

ax.contourf(X, Y, feasible.astype(float),
            levels=[0.5, 1.5], colors=['#d4edda'], alpha=0.4, zorder=0)
ax.contour(X, Y, feasible.astype(float),
           levels=[0.5], colors=[TOL['green']], linewidths=0.5, alpha=0.3)
```

---

## 13. Saving

`savefig.dpi=300` and `savefig.bbox='tight'` are already in `IEEE_RC`, so:

```python
plt.savefig('topic_what_is_plotted.png')   # dpi and bbox applied automatically
plt.show()
```

Naming convention: `{topic}_{what_is_plotted}.png`

---

## 14. Complete Single-Plot Template

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

TOL = {
    'blue': '#4477AA', 'red': '#EE6677', 'green': '#228833',
    'yellow': '#CCBB44', 'cyan': '#66CCEE', 'purple': '#AA3377',
    'grey': '#BBBBBB',
}

IEEE_RC = {
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'legend.framealpha': 0.85,
    'legend.edgecolor': '0.8', 'legend.fancybox': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'grid.linewidth': 0.5, 'grid.alpha': 0.7,
}

sns.set_theme(style="ticks", font_scale=1.0)

with plt.rc_context(IEEE_RC):
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(x, y, s=8, color=TOL['blue'], label='Data', zorder=2)
    ax.plot(x_opt, y_opt, marker='*', markersize=18,
            color=TOL['red'], markeredgecolor=TOL['red'],
            zorder=3, label='Optimal')

    ax.set_xlabel(r'$x$-axis label  (units)')
    ax.set_ylabel(r'$y$-axis label  (units)')
    ax.set_title(r'Plot Title' + '\n' + r'Parameter values here')
    ax.legend(loc='best')
    ax.grid(True)

    fig.tight_layout()
    plt.savefig('output_name.png')
    plt.show()
```

---

## 15. Complete Two-Panel Template

```python
sns.set_theme(style="ticks", font_scale=1.0)

with plt.rc_context(IEEE_RC):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel
    sc1 = ax1.scatter(x, y, c=values, cmap='viridis', s=10, zorder=2)
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="4%", pad=0.08)
    cb1 = fig.colorbar(sc1, cax=cax1)
    cb1.set_label(r'Left Colorbar Label')
    cb1.ax.tick_params(labelsize=8)
    ax1.set_xlabel(r'$x$'); ax1.set_ylabel(r'$y$')
    ax1.set_title(r'Left Panel Title')
    ax1.grid(True)

    # Right panel
    sc2 = ax2.scatter(x, y, c=other_values, cmap='RdBu_r', s=10, zorder=2)
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="4%", pad=0.08)
    cb2 = fig.colorbar(sc2, cax=cax2)
    cb2.set_label(r'Right Colorbar Label')
    cb2.ax.tick_params(labelsize=8)
    ax2.set_xlabel(r'$x$'); ax2.set_ylabel(r'$y$')
    ax2.set_title(r'Right Panel Title')
    ax2.grid(True)

    fig.suptitle(r'Overall Figure Title', fontsize=11)
    fig.tight_layout()
    plt.savefig('two_panel_output.png')
    plt.show()
```

---

## 16. Dependencies

```
numpy
matplotlib        # includes mpl_toolkits, no separate install needed
seaborn
```
