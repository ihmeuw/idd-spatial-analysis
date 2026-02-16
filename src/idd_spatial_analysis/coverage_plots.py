"""
coverage_plots.py
=================
Summary plots for district clinic coverage curves.

Each plot function produces TWO figures (0-20 km and 0-100 km x-axis range)
and saves a companion CSV with the plotted data.

Helper
------
    parse_thresholds(df)  → (pct_cols, thresholds_km)

Plot functions
--------------
    plot_adm2_lines_by_clinic_count(df, adm1_name, ...)   Ex 1
    plot_adm1_mean_lines(df, ...)                          Ex 2
    plot_adm1_few_vs_many(df, threshold_clinics, ...)      Ex 3
    plot_adm2_lines_by_exact_count(df, adm1_name, ...)     Ex 4
    plot_adm2_lines_binned(df, adm1_name, bins, ...)       Ex 5
    plot_multipanel_adm1(df, plot_types, ...)              Ex 6
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Optional, Tuple, Dict


# =============================================================================
# Helpers
# =============================================================================

def parse_thresholds(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """Extract sorted (pct_cols, thresholds_km) from a coverage DataFrame."""
    pct_cols = sorted([c for c in df.columns if c.startswith('pct_within_')],
                      key=lambda c: float(c.replace('pct_within_','')
                                          .replace('km','').replace('p','.')))
    thresholds = [float(c.replace('pct_within_','').replace('km','').replace('p','.'))
                  for c in pct_cols]
    return pct_cols, thresholds


def _clinic_count_cmap(n_clinics_series: pd.Series, cmap_name='YlOrRd'):
    vmin, vmax = 0, max(n_clinics_series.max(), 1)
    return mcolors.Normalize(vmin=vmin, vmax=vmax), cm.get_cmap(cmap_name)


def _shade_ui(ax, thresholds, mean, lo, hi, color, alpha=0.18, **line_kw):
    ax.fill_between(thresholds, lo, hi, color=color, alpha=alpha)
    ax.plot(thresholds, mean, color=color, **line_kw)


def _style_ax(ax, xlim=None, ylim=(0, 100),
              xlabel='Distance threshold (km)',
              ylabel='% population within threshold'):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, lw=0.4, alpha=0.4, color='#cccccc')
    ax.spines[['top', 'right']].set_visible(False)


def _save_fig(fig, output_dir, stem, xlim_tag, dpi=150):
    """Save figure with xlim suffix."""
    if output_dir is not None:
        p = Path(output_dir) / f'{stem}_{xlim_tag}.png'
        fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'  Saved: {p.name}')


def _save_csv(df_out, output_dir, stem):
    """Save companion CSV."""
    if output_dir is not None:
        p = Path(output_dir) / f'{stem}.csv'
        df_out.to_csv(p, index=False, float_format='%.2f')
        print(f'  Saved: {p.name}')


def _two_figs(make_fn, stem, output_dir, dpi=150):
    """
    Call make_fn(ax, xlim) twice — once for 0-20 km, once for 0-100 km.
    make_fn should draw onto ax and apply _style_ax itself.
    Returns list of (fig, ax) pairs.
    """
    results = []
    for xlim, tag in [((0, 20), '0_20km'), ((0, 100), '0_100km')]:
        fig, ax = plt.subplots(figsize=(8, 5))
        make_fn(ax, xlim)
        fig.tight_layout()
        _save_fig(fig, output_dir, stem, tag, dpi)
        results.append((fig, ax))
    return results


def _ncols_nrows(n: int) -> Tuple[int, int]:
    """Choose a compact grid layout for n panels."""
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    return ncols, nrows


# =============================================================================
# Clinic count bins (default)
# =============================================================================

DEFAULT_BINS = [
    (0,  0,   'No clinics',  '#d73027'),
    (1,  4,   '1–4 clinics', '#fc8d59'),
    (5,  9,   '5–9 clinics', '#74add1'),
    (10, 9999,'10+ clinics', '#313695'),
]


# =============================================================================
# Example 1: one line per district, colored by n_clinics
# =============================================================================

def plot_adm2_lines_by_clinic_count(
    df: pd.DataFrame,
    adm1_name: str,
    pct_cols: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    cmap_name: str = 'YlOrRd',
    alpha: float = 0.6,
    lw: float = 1.0,
    output_dir=None,
    dpi: int = 150,
):
    """Ex 1: one line per district in adm1_name, colored by n_clinics."""
    if pct_cols is None or thresholds is None:
        pct_cols, thresholds = parse_thresholds(df)

    sub = df[df['ADM1_NAME'] == adm1_name].copy()
    if sub.empty:
        raise ValueError(f"No data for province '{adm1_name}'")

    norm, cmap = _clinic_count_cmap(sub['n_clinics'], cmap_name)
    stem = f'ex1_adm2_lines_{adm1_name.replace(" ", "_")}'

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_rows = []
    for _, row in sub.iterrows():
        for col, thresh in zip(pct_cols, thresholds):
            csv_rows.append({
                'ADM2_NAME': row['ADM2_NAME'],
                'n_clinics': row['n_clinics'],
                'threshold_km': thresh,
                'pct_within': row[col],
            })
    _save_csv(pd.DataFrame(csv_rows), output_dir, stem)

    # ── Figures ───────────────────────────────────────────────────────────────
    def draw(ax, xlim):
        for _, row in sub.iterrows():
            color = cmap(norm(row['n_clinics']))
            ax.plot(thresholds, row[pct_cols].values.astype(float),
                    color=color, alpha=alpha, lw=lw)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig = ax.get_figure()
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label('Number of clinics', fontsize=9)
        ax.set_title(f'{adm1_name} — coverage by district', fontsize=11)
        _style_ax(ax, xlim=xlim)

    return _two_figs(draw, stem, output_dir, dpi)


# =============================================================================
# Example 2: mean line per province
# =============================================================================

def plot_adm1_mean_lines(
    df: pd.DataFrame,
    pct_cols: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    adm1_colors: Optional[Dict[str, str]] = None,
    lw: float = 1.8,
    output_dir=None,
    dpi: int = 150,
):
    """Ex 2: one mean coverage line per province."""
    if pct_cols is None or thresholds is None:
        pct_cols, thresholds = parse_thresholds(df)

    adm1_names = sorted(df['ADM1_NAME'].unique())
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = adm1_colors or {n: default_colors[i % len(default_colors)]
                                 for i, n in enumerate(adm1_names)}
    stem = 'ex2_adm1_mean_lines'

    # ── CSV ───────────────────────────────────────────────────────────────────
    rows = []
    for adm1 in adm1_names:
        sub = df[df['ADM1_NAME'] == adm1]
        for col, thresh in zip(pct_cols, thresholds):
            rows.append({'ADM1_NAME': adm1, 'threshold_km': thresh,
                         'mean_pct': sub[col].mean()})
    _save_csv(pd.DataFrame(rows), output_dir, stem)

    # ── Figures ───────────────────────────────────────────────────────────────
    def draw(ax, xlim):
        for adm1 in adm1_names:
            sub = df[df['ADM1_NAME'] == adm1]
            mean = sub[pct_cols].mean().values
            ax.plot(thresholds, mean, color=color_map[adm1], lw=lw, label=adm1)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_title('Mean coverage by province', fontsize=11)
        _style_ax(ax, xlim=xlim)

    return _two_figs(draw, stem, output_dir, dpi)


# =============================================================================
# Example 3: few vs many clinics, one line pair per province
# =============================================================================

def plot_adm1_few_vs_many(
    df: pd.DataFrame,
    threshold_clinics: int = 5,
    pct_cols: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    few_color: str = '#d73027',
    many_color: str = '#313695',
    lw: float = 1.8,
    output_dir=None,
    dpi: int = 150,
):
    """Ex 3: for each province, two lines — few vs many clinics."""
    if pct_cols is None or thresholds is None:
        pct_cols, thresholds = parse_thresholds(df)

    adm1_names = sorted(df['ADM1_NAME'].unique())
    linestyles = ['-', '--', ':', '-.', (0,(3,1,1,1)), (0,(5,2))]
    stem = f'ex3_few_vs_many_t{threshold_clinics}'

    # ── CSV ───────────────────────────────────────────────────────────────────
    rows = []
    for adm1 in adm1_names:
        sub = df[df['ADM1_NAME'] == adm1]
        for col, thresh in zip(pct_cols, thresholds):
            few_mean  = sub[sub['n_clinics'] <  threshold_clinics][col].mean()
            many_mean = sub[sub['n_clinics'] >= threshold_clinics][col].mean()
            rows.append({'ADM1_NAME': adm1, 'threshold_km': thresh,
                         'few_mean_pct': few_mean, 'many_mean_pct': many_mean})
    _save_csv(pd.DataFrame(rows), output_dir, stem)

    # ── Figures ───────────────────────────────────────────────────────────────
    def draw(ax, xlim):
        for i, adm1 in enumerate(adm1_names):
            sub = df[df['ADM1_NAME'] == adm1]
            ls = linestyles[i % len(linestyles)]
            few  = sub[sub['n_clinics'] <  threshold_clinics][pct_cols].mean().values
            many = sub[sub['n_clinics'] >= threshold_clinics][pct_cols].mean().values
            ax.plot(thresholds, few,  color=few_color,  lw=lw, ls=ls)
            ax.plot(thresholds, many, color=many_color, lw=lw, ls=ls)

        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0],[0], color=few_color,  lw=2,
                   label=f'<{threshold_clinics} clinics'),
            Line2D([0],[0], color=many_color, lw=2,
                   label=f'≥{threshold_clinics} clinics'),
        ] + [Line2D([0],[0], color='#555', lw=1.5,
                    ls=linestyles[i % len(linestyles)], label=adm1)
             for i, adm1 in enumerate(adm1_names)]
        ax.legend(handles=legend_elems, fontsize=7, framealpha=0.7, ncol=2)
        ax.set_title(
            f'<{threshold_clinics} vs ≥{threshold_clinics} clinics by province',
            fontsize=11)
        _style_ax(ax, xlim=xlim)

    return _two_figs(draw, stem, output_dir, dpi)


# =============================================================================
# Example 4: mean + UI per exact clinic count, one province
# =============================================================================

def plot_adm2_lines_by_exact_count(
    df: pd.DataFrame,
    adm1_name: str,
    pct_cols: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    max_count: Optional[int] = None,
    cmap_name: str = 'viridis',
    ci: Tuple[float, float] = (2.5, 97.5),
    lw: float = 1.6,
    output_dir=None,
    dpi: int = 150,
):
    """Ex 4: mean + shaded UI per exact n_clinics value, one province."""
    if pct_cols is None or thresholds is None:
        pct_cols, thresholds = parse_thresholds(df)

    sub = df[df['ADM1_NAME'] == adm1_name].copy()
    if sub.empty:
        raise ValueError(f"No data for province '{adm1_name}'")

    counts = sorted(sub['n_clinics'].unique())
    if max_count is not None:
        counts = [c for c in counts if c <= max_count]

    norm  = mcolors.Normalize(vmin=min(counts), vmax=max(counts))
    cmap  = cm.get_cmap(cmap_name)
    stem  = f'ex4_exact_count_{adm1_name.replace(" ", "_")}'

    # ── CSV ───────────────────────────────────────────────────────────────────
    rows = []
    for n in counts:
        grp = sub[sub['n_clinics'] == n][pct_cols].values.astype(float)
        if grp.shape[0] == 0:
            continue
        for col, thresh in zip(pct_cols, thresholds):
            col_vals = sub[sub['n_clinics'] == n][col].values.astype(float)
            rows.append({
                'n_clinics': n, 'threshold_km': thresh,
                'mean_pct':   np.nanmean(col_vals),
                f'p{ci[0]}_pct': np.nanpercentile(col_vals, ci[0]),
                f'p{ci[1]}_pct': np.nanpercentile(col_vals, ci[1]),
                'n_district':  grp.shape[0],
            })
    _save_csv(pd.DataFrame(rows), output_dir, stem)

    # ── Figures ───────────────────────────────────────────────────────────────
    def draw(ax, xlim):
        for n in counts:
            grp = sub[sub['n_clinics'] == n][pct_cols].values.astype(float)
            if grp.shape[0] == 0:
                continue
            mean = np.nanmean(grp, axis=0)
            lo   = np.nanpercentile(grp, ci[0], axis=0)
            hi   = np.nanpercentile(grp, ci[1], axis=0)
            _shade_ui(ax, thresholds, mean, lo, hi, cmap(norm(n)), lw=lw,
                      label=f'{n} clinic{"s" if n!=1 else ""} (n={grp.shape[0]})')
        ax.legend(fontsize=7, framealpha=0.7, ncol=2)
        ax.set_title(
            f'{adm1_name} — mean ± {ci[0]}–{ci[1]}th pct by exact clinic count',
            fontsize=10)
        _style_ax(ax, xlim=xlim)

    return _two_figs(draw, stem, output_dir, dpi)


# =============================================================================
# Example 5: mean + UI per clinic count bin
# =============================================================================

def plot_adm2_lines_binned(
    df: pd.DataFrame,
    adm1_name: Optional[str] = None,
    bins: Optional[List[Tuple]] = None,
    pct_cols: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    ci: Tuple[float, float] = (2.5, 97.5),
    lw: float = 1.8,
    output_dir=None,
    dpi: int = 150,
):
    """Ex 5: mean + shaded UI per clinic count bin. adm1_name=None → all."""
    if pct_cols is None or thresholds is None:
        pct_cols, thresholds = parse_thresholds(df)
    if bins is None:
        bins = DEFAULT_BINS

    sub  = df if adm1_name is None else df[df['ADM1_NAME'] == adm1_name].copy()
    tag  = adm1_name or 'all'
    stem = f'ex5_binned_{tag.replace(" ", "_")}'

    # ── CSV ───────────────────────────────────────────────────────────────────
    rows = []
    for lo, hi, label, color in bins:
        grp = sub[(sub['n_clinics'] >= lo) & (sub['n_clinics'] <= hi)]
        if grp.empty:
            continue
        for col, thresh in zip(pct_cols, thresholds):
            vals = grp[col].values.astype(float)
            rows.append({
                'bin_label': label, 'threshold_km': thresh,
                'mean_pct':  np.nanmean(vals),
                f'p{ci[0]}_pct': np.nanpercentile(vals, ci[0]),
                f'p{ci[1]}_pct': np.nanpercentile(vals, ci[1]),
                'n_district':  len(grp),
            })
    _save_csv(pd.DataFrame(rows), output_dir, stem)

    # ── Figures ───────────────────────────────────────────────────────────────
    def draw(ax, xlim):
        for lo, hi, label, color in bins:
            grp = sub[(sub['n_clinics'] >= lo) & (sub['n_clinics'] <= hi)]
            if grp.empty:
                continue
            arr  = grp[pct_cols].values.astype(float)
            mean = np.nanmean(arr, axis=0)
            lo_  = np.nanpercentile(arr, ci[0], axis=0)
            hi_  = np.nanpercentile(arr, ci[1], axis=0)
            _shade_ui(ax, thresholds, mean, lo_, hi_, color, lw=lw,
                      label=f'{label} (n={len(grp)})')
        ax.legend(fontsize=8, framealpha=0.7)
        title = f'{adm1_name} — ' if adm1_name else 'All district — '
        ax.set_title(
            title + f'mean ± {ci[0]}–{ci[1]}th pct by clinic count bin',
            fontsize=10)
        _style_ax(ax, xlim=xlim)

    return _two_figs(draw, stem, output_dir, dpi)


# =============================================================================
# Example 6: multipanel — one panel per province
# =============================================================================

def plot_multipanel_adm1(
    df: pd.DataFrame,
    plot_types: List[str] = ('ex1', 'ex4', 'ex5'),
    pct_cols: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
    bins: Optional[List[Tuple]] = None,
    ci: Tuple[float, float] = (2.5, 97.5),
    panel_size: Tuple = (4, 3),
    output_dir=None,
    dpi: int = 150,
):
    """
    Ex 6: multipanel, one panel per province.
    plot_types: subset of ('ex1', 'ex4', 'ex5')
    Layout is computed automatically from number of provinces.
    Produces two figures per plot_type (0-20 km and 0-100 km).
    """
    if pct_cols is None or thresholds is None:
        pct_cols, thresholds = parse_thresholds(df)
    if bins is None:
        bins = DEFAULT_BINS

    adm1_names = sorted(df['ADM1_NAME'].unique())
    ncols, nrows = _ncols_nrows(len(adm1_names))

    all_results = {}

    for plot_type in plot_types:
        stem = f'ex6_multipanel_{plot_type}'

        # ── CSV ───────────────────────────────────────────────────────────────
        csv_rows = []

        for xlim, tag in [((0, 20), '0_20km'), ((0, 100), '0_100km')]:
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(panel_size[0]*ncols, panel_size[1]*nrows),
                squeeze=False,
            )
            axes_flat = axes.flatten()

            for i, adm1 in enumerate(adm1_names):
                ax  = axes_flat[i]
                sub = df[df['ADM1_NAME'] == adm1]

                if plot_type == 'ex1':
                    norm, cmap = _clinic_count_cmap(sub['n_clinics'])
                    for _, row in sub.iterrows():
                        ax.plot(thresholds,
                                row[pct_cols].values.astype(float),
                                color=cmap(norm(row['n_clinics'])),
                                alpha=0.5, lw=0.8)
                    if tag == '0_20km':  # collect CSV once
                        for _, row in sub.iterrows():
                            for col, thresh in zip(pct_cols, thresholds):
                                csv_rows.append({
                                    'ADM1_NAME': adm1,
                                    'ADM2_NAME': row['ADM2_NAME'],
                                    'n_clinics': row['n_clinics'],
                                    'threshold_km': thresh,
                                    'pct_within': row[col],
                                })

                elif plot_type == 'ex4':
                    counts = sorted(sub['n_clinics'].unique())
                    n_norm = mcolors.Normalize(
                        vmin=min(counts) if counts else 0,
                        vmax=max(counts) if counts else 1)
                    cmap4 = cm.get_cmap('viridis')
                    for n in counts:
                        grp = sub[sub['n_clinics']==n][pct_cols].values.astype(float)
                        if grp.shape[0] == 0:
                            continue
                        mean = np.nanmean(grp, axis=0)
                        lo_  = np.nanpercentile(grp, ci[0], axis=0)
                        hi_  = np.nanpercentile(grp, ci[1], axis=0)
                        _shade_ui(ax, thresholds, mean, lo_, hi_,
                                  cmap4(n_norm(n)), lw=1.2)
                        if tag == '0_20km':
                            for col, thresh in zip(pct_cols, thresholds):
                                vals = sub[sub['n_clinics']==n][col].values.astype(float)
                                csv_rows.append({
                                    'ADM1_NAME': adm1, 'n_clinics': n,
                                    'threshold_km': thresh,
                                    'mean_pct': np.nanmean(vals),
                                    f'p{ci[0]}_pct': np.nanpercentile(vals, ci[0]),
                                    f'p{ci[1]}_pct': np.nanpercentile(vals, ci[1]),
                                })

                elif plot_type == 'ex5':
                    for lo, hi, label, color in bins:
                        grp = sub[(sub['n_clinics']>=lo) & (sub['n_clinics']<=hi)]
                        if grp.empty:
                            continue
                        arr  = grp[pct_cols].values.astype(float)
                        mean = np.nanmean(arr, axis=0)
                        lo_  = np.nanpercentile(arr, ci[0], axis=0)
                        hi_  = np.nanpercentile(arr, ci[1], axis=0)
                        _shade_ui(ax, thresholds, mean, lo_, hi_, color,
                                  lw=1.2, label=label)
                        if tag == '0_20km':
                            for col, thresh in zip(pct_cols, thresholds):
                                vals = grp[col].values.astype(float)
                                csv_rows.append({
                                    'ADM1_NAME': adm1, 'bin_label': label,
                                    'threshold_km': thresh,
                                    'mean_pct': np.nanmean(vals),
                                    f'p{ci[0]}_pct': np.nanpercentile(vals, ci[0]),
                                    f'p{ci[1]}_pct': np.nanpercentile(vals, ci[1]),
                                    'n_district': len(grp),
                                })

                ax.set_title(adm1, fontsize=8, pad=3)
                _style_ax(ax, xlim=xlim)
                ax.tick_params(labelsize=7)
                ax.set_xlabel(''); ax.set_ylabel('')

            # Hide unused panels
            for j in range(len(adm1_names), len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.supxlabel('Distance threshold (km)', fontsize=9, y=0.01)
            fig.supylabel('% population within threshold', fontsize=9, x=0.01)

            if plot_type == 'ex5':
                from matplotlib.patches import Patch
                handles = [Patch(facecolor=c, label=l) for _,_,l,c in bins]
                fig.legend(handles=handles, loc='lower right', fontsize=7,
                           framealpha=0.8, ncol=len(bins))

            type_labels = {'ex1': 'lines by clinic count',
                           'ex4': 'mean ± UI by exact count',
                           'ex5': 'mean ± UI by bin'}
            fig.suptitle(f'Coverage curves — {type_labels[plot_type]}',
                         fontsize=11, y=1.01)
            fig.tight_layout()
            _save_fig(fig, output_dir, stem, tag, dpi)

        _save_csv(pd.DataFrame(csv_rows), output_dir, stem)
        all_results[plot_type] = fig, axes

    return all_results
