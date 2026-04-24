import csv
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Basic IO / numeric utils
# -------------------------
def to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'detach') and hasattr(x, 'cpu') and hasattr(x, 'numpy'):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    try:
        return np.asarray(x)
    except Exception:
        return None



def load_piece(path: Path) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.lib.npyio.NpzFile):
        return {k: obj[k] for k in obj.files}

    if isinstance(obj, np.ndarray) and obj.shape == ():
        item = obj.item()
        if isinstance(item, dict):
            return item
        return {'value': item}

    if isinstance(obj, np.ndarray):
        return {'array': obj}

    return {'value': obj}



def flatten_numeric(x: Any) -> Optional[np.ndarray]:
    arr = to_numpy(x)
    if arr is None:
        return None

    if arr.dtype == object:
        vals = []
        
        for v in arr.reshape(-1):
            sub = to_numpy(v)
            if sub is not None and np.issubdtype(sub.dtype, np.number):
                vals.append(sub.reshape(-1))
        if not vals:
            return None
        arr = np.concatenate(vals)

    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr.reshape(-1)



def safe_stats(arr: Optional[np.ndarray]) -> Dict[str, Any]:
    if arr is None or arr.size == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'p50': np.nan,
            'p90': np.nan,
            'p95': np.nan,
            'max': np.nan,
        }
    arr = np.asarray(arr, dtype=float).reshape(-1)
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(np.max(arr)),
    }



def infer_object_scale(path: Path) -> Tuple[str, str]:
    name = path.stem
    m = re.match(r'(scale\d+)_grasp$', name)
    scale = m.group(1) if m else ''
    obj_code = path.parent.parent.name if path.parent.name == 'floating' else path.parent.name
    return obj_code, scale



def to_key(obj_code: str, scale: str) -> str:
    return f'{obj_code}/{scale}' if scale else obj_code


# -------------------------
# Summarize one file / root
# -------------------------
def summarize_piece(path: Path, dist_thr: float, grasp_thr: Optional[float], label: str) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    data = load_piece(path)

    dist = flatten_numeric(data.get('dist_error'))
    grasp = flatten_numeric(data.get('grasp_error'))
    force = to_numpy(data.get('contact_force'))

    obj_code, scale = infer_object_scale(path)
    row: Dict[str, Any] = {
        'source_label': label,
        'file': str(path),
        'object_code': obj_code,
        'scale': scale,
        'object_scale': to_key(obj_code, scale),
    }

    dist_s = safe_stats(dist)
    for k, v in dist_s.items():
        row[f'dist_{k}'] = v
    row['succ_dist_thr_count'] = int(np.sum(dist < dist_thr)) if dist is not None else 0
    row['succ_dist_thr_ratio'] = float(np.mean(dist < dist_thr)) if dist is not None and dist.size > 0 else np.nan

    grasp_s = safe_stats(grasp)
    for k, v in grasp_s.items():
        row[f'grasp_{k}'] = v
    if grasp_thr is not None:
        row['succ_grasp_thr_count'] = int(np.sum(grasp < grasp_thr)) if grasp is not None else 0
        row['succ_grasp_thr_ratio'] = float(np.mean(grasp < grasp_thr)) if grasp is not None and grasp.size > 0 else np.nan

    if dist is not None and grasp_thr is not None and grasp is not None and dist.size > 0 and grasp.size > 0:
        n = min(dist.size, grasp.size)
        row['succ_joint_thr_ratio'] = float(np.mean((dist[:n] < dist_thr) & (grasp[:n] < grasp_thr)))
    else:
        row['succ_joint_thr_ratio'] = np.nan

    if force is not None and np.issubdtype(np.asarray(force).dtype, np.number):
        f = np.asarray(force)
        row['contact_force_shape'] = 'x'.join(map(str, f.shape))
        mags = np.linalg.norm(f, axis=-1) if f.ndim >= 2 and f.shape[-1] == 3 else f.reshape(-1)
        force_s = safe_stats(mags.reshape(-1))
        for k, v in force_s.items():
            row[f'force_{k}'] = v
    else:
        row['contact_force_shape'] = ''
        for k in ['count', 'mean', 'std', 'min', 'p50', 'p90', 'p95', 'max']:
            row[f'force_{k}'] = 0 if k == 'count' else np.nan

    return row, dist, grasp



def summarize_root(root: Path, glob_pat: str, dist_thr: float, grasp_thr: Optional[float], label: str):
    files = sorted(root.glob(glob_pat))
    if not files:
        raise FileNotFoundError(f'No files found under: {root} with glob: {glob_pat}')

    rows: List[Dict[str, Any]] = []
    bad: List[Dict[str, Any]] = []
    dist_all: List[np.ndarray] = []
    grasp_all: List[np.ndarray] = []

    for p in files:
        try:
            row, dist, grasp = summarize_piece(p, dist_thr, grasp_thr, label)
            rows.append(row)
            if dist is not None and dist.size > 0:
                dist_all.append(dist.reshape(-1))
            if grasp is not None and grasp.size > 0:
                grasp_all.append(grasp.reshape(-1))
        except Exception as e:
            bad.append({'file': str(p), 'error': str(e), 'source_label': label})

    dist_concat = np.concatenate(dist_all) if dist_all else None
    grasp_concat = np.concatenate(grasp_all) if grasp_all else None
    return rows, bad, dist_concat, grasp_concat


# -------------------------
# Aggregation / CSV helpers
# -------------------------
def is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not np.isnan(x)



def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        'dist_mean', 'dist_p50', 'dist_p90', 'dist_max',
        'grasp_mean', 'grasp_p50', 'grasp_p90', 'grasp_max',
        'succ_dist_thr_ratio', 'succ_grasp_thr_ratio', 'succ_joint_thr_ratio',
        'force_mean', 'force_p90'
    ]
    out = {'num_files': len(rows)}
    for key in keys:
        vals = [r[key] for r in rows if key in r and is_number(r[key])]
        out[key] = float(np.mean(vals)) if vals else np.nan
    return out



def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



def group_object_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[r['object_scale']].append(r)

    out = []
    metrics = [
        'dist_mean', 'dist_p50', 'dist_p90', 'dist_max',
        'grasp_mean', 'grasp_p50', 'grasp_p90', 'grasp_max',
        'succ_dist_thr_ratio', 'succ_grasp_thr_ratio', 'succ_joint_thr_ratio',
        'force_mean', 'force_p90'
    ]
    for object_scale, items in buckets.items():
        first = items[0]
        row = {
            'source_label': first['source_label'],
            'object_scale': object_scale,
            'object_code': first['object_code'],
            'scale': first['scale'],
            'num_files': len(items),
        }
        for m in metrics:
            vals = [x[m] for x in items if m in x and is_number(x[m])]
            row[m] = float(np.mean(vals)) if vals else np.nan
        out.append(row)

    out.sort(key=lambda x: x['object_scale'])
    return out



def merge_object_comparison(obj_a: List[Dict[str, Any]], obj_b: List[Dict[str, Any]], label_a: str, label_b: str) -> List[Dict[str, Any]]:
    map_a = {r['object_scale']: r for r in obj_a}
    map_b = {r['object_scale']: r for r in obj_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))

    merged = []
    metrics = [
        'dist_mean', 'dist_p90', 'grasp_mean', 'grasp_p90',
        'succ_dist_thr_ratio', 'succ_grasp_thr_ratio', 'succ_joint_thr_ratio'
    ]
    for k in common:
        ra = map_a[k]
        rb = map_b[k]
        row = {
            'object_scale': k,
            'object_code': ra['object_code'],
            'scale': ra['scale'],
        }
        for m in metrics:
            row[f'{label_a}_{m}'] = ra.get(m, np.nan)
            row[f'{label_b}_{m}'] = rb.get(m, np.nan)
            a = row[f'{label_a}_{m}']
            b = row[f'{label_b}_{m}']
            row[f'delta_{label_b}_minus_{label_a}_{m}'] = (b - a) if is_number(a) and is_number(b) else np.nan
        merged.append(row)
    return merged


# -------------------------
# Plotting helpers
# -------------------------
def setup_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.bbox': 'tight',
        'savefig.dpi': 220,
    })



def maybe_downsample(arr: Optional[np.ndarray], max_points: int = 200000) -> Optional[np.ndarray]:
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if arr.size <= max_points:
        return arr
    idx = np.linspace(0, arr.size - 1, max_points).astype(int)
    return arr[idx]



def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(arr.reshape(-1))
    y = np.arange(1, x.size + 1) / x.size
    return x, y



def add_panel_tag(ax, tag: str):
    ax.text(-0.12, 1.05, tag, transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')



def choose_success_metric(has_grasp_thr: bool) -> str:
    return 'succ_joint_thr_ratio' if has_grasp_thr else 'succ_dist_thr_ratio'



def plot_panel(
    dist_a: Optional[np.ndarray],
    dist_b: Optional[np.ndarray],
    grasp_a: Optional[np.ndarray],
    grasp_b: Optional[np.ndarray],
    obj_cmp: List[Dict[str, Any]],
    label_a: str,
    label_b: str,
    out_png: Path,
    grasp_thr: Optional[float],
    max_objects: int = 12,
):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    ax1, ax2, ax3, ax4 = axes.reshape(-1)

    color_a = '#4C78A8'
    color_b = '#E45756'

    # A: dist ECDF
    if dist_a is not None:
        xa, ya = ecdf(maybe_downsample(dist_a))
        ax1.plot(xa, ya, label=label_a, linewidth=2.0, color=color_a)
    if dist_b is not None:
        xb, yb = ecdf(maybe_downsample(dist_b))
        ax1.plot(xb, yb, label=label_b, linewidth=2.0, color=color_b)
    ax1.set_title('Distribution of dist_error')
    ax1.set_xlabel('dist_error')
    ax1.set_ylabel('ECDF')
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    add_panel_tag(ax1, 'A')

    # B: grasp ECDF
    if grasp_a is not None:
        xa, ya = ecdf(maybe_downsample(grasp_a))
        ax2.plot(xa, ya, label=label_a, linewidth=2.0, color=color_a)
    if grasp_b is not None:
        xb, yb = ecdf(maybe_downsample(grasp_b))
        ax2.plot(xb, yb, label=label_b, linewidth=2.0, color=color_b)
    ax2.set_title('Distribution of grasp_error')
    ax2.set_xlabel('grasp_error')
    ax2.set_ylabel('ECDF')
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    add_panel_tag(ax2, 'B')

    # C/D: object-level grouped bars on common objects
    if obj_cmp:
        diff_key = f'delta_{label_b}_minus_{label_a}_dist_mean'
        ranked = sorted(
            obj_cmp,
            key=lambda r: abs(r.get(diff_key, np.nan)) if is_number(r.get(diff_key, np.nan)) else -1,
            reverse=True,
        )
        ranked = ranked[:max_objects]
        names = [r['object_scale'] for r in ranked]
        x = np.arange(len(names))
        width = 0.38

        a_vals = [r.get(f'{label_a}_dist_mean', np.nan) for r in ranked]
        b_vals = [r.get(f'{label_b}_dist_mean', np.nan) for r in ranked]
        ax3.bar(x - width / 2, a_vals, width=width, label=label_a, color=color_a, alpha=0.9)
        ax3.bar(x + width / 2, b_vals, width=width, label=label_b, color=color_b, alpha=0.9)
        ax3.set_title('Object-level mean dist_error')
        ax3.set_ylabel('mean dist_error')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=40, ha='right')
        ax3.grid(axis='y', alpha=0.25)
        ax3.legend(frameon=False)
        add_panel_tag(ax3, 'C')

        succ_key = choose_success_metric(grasp_thr is not None)
        a_vals = [r.get(f'{label_a}_{succ_key}', np.nan) for r in ranked]
        b_vals = [r.get(f'{label_b}_{succ_key}', np.nan) for r in ranked]
        ax4.bar(x - width / 2, a_vals, width=width, label=label_a, color=color_a, alpha=0.9)
        ax4.bar(x + width / 2, b_vals, width=width, label=label_b, color=color_b, alpha=0.9)
        title = 'Object-level joint success rate' if grasp_thr is not None else 'Object-level dist success rate'
        ax4.set_title(title)
        ax4.set_ylabel('success ratio')
        ax4.set_ylim(0.0, 1.0)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=40, ha='right')
        ax4.grid(axis='y', alpha=0.25)
        ax4.legend(frameon=False)
        add_panel_tag(ax4, 'D')
    else:
        for ax, tag, title in [(ax3, 'C', 'Object-level mean dist_error'), (ax4, 'D', 'Object-level success rate')]:
            ax.text(0.5, 0.5, 'No common objects to compare', ha='center', va='center')
            ax.set_title(title)
            add_panel_tag(ax, tag)
            ax.set_axis_off()

    fig.suptitle(f'BODex comparison: {label_a} vs {label_b}')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)



def plot_single_root_panel(
    dist_all: Optional[np.ndarray],
    grasp_all: Optional[np.ndarray],
    object_rows: List[Dict[str, Any]],
    label: str,
    out_png: Path,
    grasp_thr: Optional[float],
    max_objects: int = 12,
):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6))
    ax1, ax2, ax3, ax4 = axes.reshape(-1)
    color = '#4C78A8'

    if dist_all is not None:
        x, y = ecdf(maybe_downsample(dist_all))
        ax1.plot(x, y, linewidth=2.0, color=color)
    ax1.set_title('Distribution of dist_error')
    ax1.set_xlabel('dist_error')
    ax1.set_ylabel('ECDF')
    ax1.grid(alpha=0.25)
    add_panel_tag(ax1, 'A')

    if grasp_all is not None:
        x, y = ecdf(maybe_downsample(grasp_all))
        ax2.plot(x, y, linewidth=2.0, color=color)
    ax2.set_title('Distribution of grasp_error')
    ax2.set_xlabel('grasp_error')
    ax2.set_ylabel('ECDF')
    ax2.grid(alpha=0.25)
    add_panel_tag(ax2, 'B')

    ranked = sorted(object_rows, key=lambda r: r.get('dist_mean', np.nan) if is_number(r.get('dist_mean', np.nan)) else -1, reverse=True)[:max_objects]
    names = [r['object_scale'] for r in ranked]
    x = np.arange(len(names))

    ax3.bar(x, [r.get('dist_mean', np.nan) for r in ranked], color=color, alpha=0.9)
    ax3.set_title('Object-level mean dist_error')
    ax3.set_ylabel('mean dist_error')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=40, ha='right')
    ax3.grid(axis='y', alpha=0.25)
    add_panel_tag(ax3, 'C')

    succ_key = choose_success_metric(grasp_thr is not None)
    ax4.bar(x, [r.get(succ_key, np.nan) for r in ranked], color=color, alpha=0.9)
    ax4.set_title('Object-level success rate')
    ax4.set_ylabel('success ratio')
    ax4.set_ylim(0.0, 1.0)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=40, ha='right')
    ax4.grid(axis='y', alpha=0.25)
    add_panel_tag(ax4, 'D')

    fig.suptitle(f'BODex summary: {label}')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


# -------------------------
# Text report
# -------------------------
def fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return 'nan'
        return f'{float(v):.{digits}f}'
    return str(v)



def build_text_report(summary_a: Dict[str, Any], summary_b: Optional[Dict[str, Any]], label_a: str, label_b: Optional[str], grasp_thr: Optional[float]) -> str:
    lines = []
    lines.append('===== OVERALL SUMMARY =====')
    lines.append(f'[{label_a}]')
    for k, v in summary_a.items():
        lines.append(f'  {k}: {fmt(v)}')

    if summary_b is not None and label_b is not None:
        lines.append(f'[{label_b}]')
        for k, v in summary_b.items():
            lines.append(f'  {k}: {fmt(v)}')

        lines.append('')
        lines.append('===== COMPARISON =====')
        for key in ['dist_mean', 'dist_p90', 'grasp_mean', 'grasp_p90', 'succ_dist_thr_ratio']:
            a = summary_a.get(key, np.nan)
            b = summary_b.get(key, np.nan)
            delta = b - a if is_number(a) and is_number(b) else np.nan
            lines.append(f'  {key}: {label_b} - {label_a} = {fmt(delta)}')

        if grasp_thr is not None:
            for key in ['succ_grasp_thr_ratio', 'succ_joint_thr_ratio']:
                a = summary_a.get(key, np.nan)
                b = summary_b.get(key, np.nan)
                delta = b - a if is_number(a) and is_number(b) else np.nan
                lines.append(f'  {key}: {label_b} - {label_a} = {fmt(delta)}')

    return '\n'.join(lines)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='Summarize and visualize BODex graspdata outputs')
    parser.add_argument('--root-a', required=True, help='First graspdata root directory')
    parser.add_argument('--root-b', default=None, help='Second graspdata root directory for comparison')
    parser.add_argument('--label-a', default='shadowhand')
    parser.add_argument('--label-b', default='hand_f')
    parser.add_argument('--glob', default='**/*grasp.npy', help='File glob under root')
    parser.add_argument('--dist-thr', type=float, default=0.1, help='Success threshold for dist_error')
    parser.add_argument('--grasp-thr', type=float, default=None, help='Optional success threshold for grasp_error')
    parser.add_argument('--outdir', default='bodex_compare_results', help='Where to save csv/figures')
    parser.add_argument('--max-objects', type=int, default=12, help='Maximum objects shown in object-level plots')
    args = parser.parse_args()

    root_a = Path(args.root_a)
    root_b = Path(args.root_b) if args.root_b else None
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows_a, bad_a, dist_a, grasp_a = summarize_root(root_a, args.glob, args.dist_thr, args.grasp_thr, args.label_a)
    obj_a = group_object_rows(rows_a)
    summary_a = aggregate_rows(rows_a)

    write_csv(rows_a, outdir / f'per_file_{args.label_a}.csv')
    write_csv(obj_a, outdir / f'per_object_{args.label_a}.csv')
    write_csv(bad_a, outdir / f'load_errors_{args.label_a}.csv')

    if root_b is None:
        plot_single_root_panel(dist_a, grasp_a, obj_a, args.label_a, outdir / f'panel_{args.label_a}.png', args.grasp_thr, args.max_objects)
        report = build_text_report(summary_a, None, args.label_a, None, args.grasp_thr)
        (outdir / 'summary.txt').write_text(report, encoding='utf-8')
        print(report)
        print(f'\nSaved results to: {outdir.resolve()}')
        return

    rows_b, bad_b, dist_b, grasp_b = summarize_root(root_b, args.glob, args.dist_thr, args.grasp_thr, args.label_b)
    obj_b = group_object_rows(rows_b)
    summary_b = aggregate_rows(rows_b)

    write_csv(rows_b, outdir / f'per_file_{args.label_b}.csv')
    write_csv(obj_b, outdir / f'per_object_{args.label_b}.csv')
    write_csv(bad_b, outdir / f'load_errors_{args.label_b}.csv')

    obj_cmp = merge_object_comparison(obj_a, obj_b, args.label_a, args.label_b)
    write_csv(obj_cmp, outdir / f'comparison_{args.label_a}_vs_{args.label_b}.csv')

    plot_panel(
        dist_a, dist_b, grasp_a, grasp_b,
        obj_cmp, args.label_a, args.label_b,
        outdir / f'panel_{args.label_a}_vs_{args.label_b}.png',
        args.grasp_thr, args.max_objects,
    )

    report = build_text_report(summary_a, summary_b, args.label_a, args.label_b, args.grasp_thr)
    (outdir / 'summary.txt').write_text(report, encoding='utf-8')
    print(report)
    print(f'\nSaved results to: {outdir.resolve()}')


if __name__ == '__main__':
    main()
