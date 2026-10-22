import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path('./muraves_chi2_scan')
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "images"
DEFAULT_DETECTOR = 'ROSSO'
CHI2_MIN = 0.0
CHI2_MAX = 12.0
CHI2_STEP = 0.1
VES_PLOT_MIN = 1.0 # start plotting chi2 cuts from this value (because blow up for very small cut)


def build_chi2_cuts():
    """Build chi2 cuts from CHI2_MIN to CHI2_MAX in steps of CHI2_STEP."""
    return np.round(np.arange(CHI2_MIN, CHI2_MAX + 0.5 * CHI2_STEP, CHI2_STEP), 1)


def find_common_working_points(input_dir):
    """Find working points for which both VES and FS parquet files exist."""
    ves_files = sorted(input_dir.glob('scan_VES_WP*.parquet'))
    common_wps = []
    for ves_file in ves_files:
        filename = ves_file.name
        if not filename.startswith('scan_VES_WP'):
            continue
        wp_string = filename[len('scan_VES_WP'):-len('.parquet')]
        try:
            wp = int(wp_string)
        except ValueError:
            continue
        fs_file = input_dir / f'scan_FS_WP{wp}.parquet'
        if fs_file.exists():
            common_wps.append(wp)
    return sorted(set(common_wps))


def get_parquet_files(input_dir, working_point):
    """Return VES and FS parquet files for a working point."""
    ves_file = input_dir / f'scan_VES_WP{working_point}.parquet'
    fs_file = input_dir / f'scan_FS_WP{working_point}.parquet'
    if not ves_file.exists():
        raise FileNotFoundError(f'VES file not found: {ves_file}')
    if not fs_file.exists():
        raise FileNotFoundError(f'FS file not found: {fs_file}')
    return (ves_file, fs_file)


def select_requested_cuts(available_cuts, requested_cuts, tolerance=1e-06):
    """
    Match requested FS chi2 cuts against the cuts available
    in the parquet.

    Floating-point differences are handled with np.isclose().
    """
    available_cuts = np.asarray(available_cuts, dtype=float)
    selected = []
    for requested in requested_cuts:
        matches = np.where(np.isclose(available_cuts, requested, atol=tolerance, rtol=0))[0]
        if len(matches) == 0:
            raise ValueError(f'Requested FS chi2 cut {requested:.1f} is not available. Available range is {available_cuts.min():.1f} to {available_cuts.max():.1f}.')
        selected.append(available_cuts[matches[0]])
    return np.array(selected)


def load_scan(parquet_file):
    """Load a chi2-scan parquet file."""
    columns = ['detector', 'data_type', 'working_point', 'chi2_cut', 'n_tracks', 'duration_seconds', 'track_rate_hz']
    df = pd.read_parquet(parquet_file, columns=columns)
    if df.empty:
        raise ValueError(f'Parquet file is empty: {parquet_file}')
    return df


def calculate_ratio_table(ves_df, fs_df):
    """
    Calculate:

        VES rate / FS rate

    for every combination of VES and FS chi2 cuts.

    Rows    = FS chi2 cuts
    Columns = VES chi2 cuts
    """
    ves_df = ves_df.sort_values('chi2_cut')
    fs_df = fs_df.sort_values('chi2_cut')
    ves_cuts = ves_df['chi2_cut'].to_numpy(dtype=float)
    fs_cuts = fs_df['chi2_cut'].to_numpy(dtype=float)
    ves_rates = ves_df['track_rate_hz'].to_numpy(dtype=float)
    fs_rates = fs_df['track_rate_hz'].to_numpy(dtype=float)
    ratio_matrix = np.full((len(fs_cuts), len(ves_cuts)), np.nan, dtype=float)
    for i, fs_rate in enumerate(fs_rates):
        if not np.isfinite(fs_rate) or fs_rate <= 0:
            continue
        ratio_matrix[i, :] = ves_rates / fs_rate
    ratio_df = pd.DataFrame(ratio_matrix, index=fs_cuts, columns=ves_cuts)
    ratio_df.index.name = 'FS_chi2_cut'
    ratio_df.columns.name = 'VES_chi2_cut'
    return ratio_df


def calculate_ratio_error_table(ves_df, fs_df):
    """Calculate propagated Poisson errors for the VES/FS rate ratio."""
    ves_df = ves_df.sort_values('chi2_cut')
    fs_df = fs_df.sort_values('chi2_cut')
    ves_cuts = ves_df['chi2_cut'].to_numpy(dtype=float)
    fs_cuts = fs_df['chi2_cut'].to_numpy(dtype=float)
    ves_rates = ves_df['track_rate_hz'].to_numpy(dtype=float)
    fs_rates = fs_df['track_rate_hz'].to_numpy(dtype=float)
    ves_counts = ves_df['n_tracks'].to_numpy(dtype=float)
    fs_counts = fs_df['n_tracks'].to_numpy(dtype=float)

    ratio_error_matrix = np.full(
        (len(fs_cuts), len(ves_cuts)),
        np.nan,
        dtype=float,
    )

    for row, (fs_rate, fs_count) in enumerate(zip(fs_rates, fs_counts)):
        valid = (
            np.isfinite(ves_rates)
            & np.isfinite(fs_rate)
            & (fs_rate > 0)
            & np.isfinite(ves_counts)
            & (ves_counts > 0)
            & np.isfinite(fs_count)
            & (fs_count > 0)
        )
        ratio_error_matrix[row, valid] = (
            (ves_rates[valid] / fs_rate)
            * np.sqrt(
                1.0 / ves_counts[valid]
                + 1.0 / fs_count
            )
        )

    return pd.DataFrame(
        ratio_error_matrix,
        index=fs_cuts,
        columns=ves_cuts,
    )


def plot_ratio(
    ratio_df,
    fs_cut,
    detector,
    working_point,
    output_dir,
    ratio_error_df=None,
):
    """
    Create one VES/FS ratio plot for a given FS chi2 cut.

    Only VES chi2 >= VES_PLOT_MIN is displayed.
    """
    fs_cut = float(fs_cut)
    fs_rows = np.where(np.isclose(ratio_df.index.to_numpy(dtype=float), fs_cut, atol=1e-06, rtol=0))[0]
    if len(fs_rows) == 0:
        raise ValueError(f'FS chi2 cut {fs_cut:.1f} not found in ratio table.')
    row = ratio_df.iloc[fs_rows[0]]
    ves_cuts = ratio_df.columns.to_numpy(dtype=float)
    ratios = row.to_numpy(dtype=float)
    ves_mask = ves_cuts >= VES_PLOT_MIN
    plot_ves_cuts = ves_cuts[ves_mask]
    plot_ratios = ratios[ves_mask]
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_errors = None
    if ratio_error_df is not None:
        error_row = ratio_error_df.iloc[fs_rows[0]].to_numpy(dtype=float)
        plot_errors = error_row[ves_mask]
    ax.errorbar(
        plot_ves_cuts,
        plot_ratios,
        yerr=plot_errors,
        fmt='o-',
        linewidth=1.5,
        markersize=3,
        capsize=2,
    )
    ax.axhline(1.0, linestyle='--', linewidth=1.0)
    ax.set_xlabel('VES $\\chi^2$ cut')
    ax.set_ylabel('VES / FS rate')
    ax.set_title(f'{detector} — WP {working_point} — FS $\\chi^2$ < {fs_cut:.1f}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(VES_PLOT_MIN, CHI2_MAX)
    fig.tight_layout()
    filename = f'ratio_VESoverFS_FSchi2_{format_chi2_cut(fs_cut)}.png'
    output_file = output_dir / filename
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file


def polynomial_prediction_error(fit_x, fit_covariance, fit_degree):
    """Return the one-sigma uncertainty of a polynomial prediction."""
    if fit_covariance is None:
        return np.full_like(fit_x, np.nan, dtype=float)
    design = np.column_stack([
        fit_x ** power for power in range(fit_degree, -1, -1)
    ])
    prediction_variance = np.einsum(
        'ij,jk,ik->i', design, fit_covariance, design,
    )
    return np.sqrt(np.maximum(prediction_variance, 0.0))


def plot_ratio_one_contour(ratio_df, detector, working_point, output_dir, tolerance=0.02, fit_degree=2):
    """
    Plot all (VES chi2 cut, FS chi2 cut) pairs for which

        VES rate / FS rate ~= 1

    within the given tolerance.

    Only VES chi2 >= VES_PLOT_MIN is displayed.
    """
    ves_cuts = ratio_df.columns.to_numpy(dtype=float)
    fs_cuts = ratio_df.index.to_numpy(dtype=float)
    ratio_matrix = ratio_df.to_numpy(dtype=float)
    mask = np.isfinite(ratio_matrix) & np.isclose(ratio_matrix, 1.0, atol=tolerance, rtol=0)
    fs_indices, ves_indices = np.where(mask)
    ves_chi2 = ves_cuts[ves_indices]
    fs_chi2 = fs_cuts[fs_indices]
    ves_mask = ves_chi2 >= VES_PLOT_MIN
    ves_chi2 = ves_chi2[ves_mask]
    fs_chi2 = fs_chi2[ves_mask]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(ves_chi2, fs_chi2, s=3, label=r'$\chi^2$ pairs')
    fit_parameters = None
    fit_covariance = None
    if len(ves_chi2) >= fit_degree + 1:
        fit_parameters = np.polyfit(ves_chi2, fs_chi2, deg=fit_degree)
        if len(ves_chi2) > fit_degree + 1:
            fit_parameters, fit_covariance = np.polyfit(
                ves_chi2,
                fs_chi2,
                deg=fit_degree,
                cov=True,
            )
        print(f'Polynomial fit parameters (degree {fit_degree}, highest power first): {fit_parameters}')
        fit_x = np.linspace(VES_PLOT_MIN, CHI2_MAX, 500)
        fit_y = np.polyval(fit_parameters, fit_x)
        fit_error = polynomial_prediction_error(fit_x, fit_covariance, fit_degree)
        equation_terms = []
        for index, parameter in enumerate(fit_parameters):
            power = fit_degree - index
            if power == 0:
                term = f'{parameter:.4g}'
            elif power == 1:
                term = f'{parameter:.4g} x'
            else:
                term = f'{parameter:.4g} x^{power}'
            equation_terms.append(term)
        fit_equation = ' + '.join(equation_terms).replace('+ -', '- ')
        ax.plot(fit_x, fit_y, linewidth=1.5, color='red', label='Polynomial fit')
        if fit_covariance is not None:
            ax.fill_between(
                fit_x,
                fit_y - fit_error,
                fit_y + fit_error,
                color='red',
                alpha=0.18,
                label='Fit $1\sigma$',
            )
        ax.text(0.03, 0.97, f'$y = {fit_equation}$', transform=ax.transAxes, verticalalignment='top', fontsize=9, bbox={'boxstyle': 'round,pad=0.4', 'facecolor': 'white', 'alpha': 0.85})
    else:
        print(f'Not enough points for a degree {fit_degree} polynomial fit: found {len(ves_chi2)} point(s).')
    #ax.plot([VES_PLOT_MIN, CHI2_MAX], [VES_PLOT_MIN, CHI2_MAX], linestyle='--', linewidth=1.0, label='$\\chi^2_{\\rm Ves} = \\chi^2_{\\rm FS}$')
    ax.set_xlabel('$\\chi^2_{Ves}$ cut')
    ax.set_ylabel('$\\chi^2_{FS}$ cut')
    ax.set_title(f'{detector} — WP {working_point}\n' rf'$\Phi_{{\rm Ves}}(\Delta\phi_{{\rm control}}, \Delta\theta_{{\rm control}}) / \Phi_{{\rm FS}}(\Delta\phi_{{\rm control}}, \Delta\theta_{{\rm control}}) = 1 \pm {tolerance}$')
    ax.set_xlim(VES_PLOT_MIN, CHI2_MAX)
    ax.set_ylim(CHI2_MIN, CHI2_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = 'ratio_equal_one.png'
    output_file = output_dir / filename
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return (output_file, fit_parameters, fit_covariance)


def interpolate_track_rate(scan_df, chi2_cut, column='track_rate_hz'):
    """
    Interpolate track rate at a chi2 cut
    from a scan dataframe.
    """
    scan_df = scan_df.sort_values('chi2_cut')
    chi2_cuts = scan_df['chi2_cut'].to_numpy(dtype=float)
    values = scan_df[column].to_numpy(dtype=float)
    finite = np.isfinite(chi2_cuts) & np.isfinite(values)
    if finite.sum() < 2:
        return np.nan
    chi2_cuts = chi2_cuts[finite]
    values = values[finite]
    if chi2_cut < chi2_cuts.min() or chi2_cut > chi2_cuts.max():
        return np.nan
    return float(np.interp(chi2_cut, chi2_cuts, values))


def interpolate_rate_error(scan_df, chi2_cut):
    """Interpolate the Poisson error of a rate at a chi2 cut."""
    rates = interpolate_track_rate(scan_df, chi2_cut, column='track_rate_hz')
    counts = interpolate_track_rate(scan_df, chi2_cut, column='n_tracks')
    durations = interpolate_track_rate(scan_df, chi2_cut, column='duration_seconds')
    if not np.isfinite(rates) or not np.isfinite(counts) or not np.isfinite(durations):
        return np.nan
    if counts <= 0 or durations <= 0:
        return np.nan
    return float(rates / np.sqrt(counts))


def polynomial_prediction_sigma(x, fit_covariance, fit_degree):
    """Return the one-sigma uncertainty of a polynomial prediction at x."""
    return float(polynomial_prediction_error(
        np.asarray([x], dtype=float), fit_covariance, fit_degree,
    )[0])


def plot_transmission_along_fit(vesuvius_ves_df, vesuvius_fs_df, fit_parameters, fit_covariance, detector, working_point, output_dir, fit_degree=2):
    """
    Plot VES/FS transmission along the fitted chi2-cut relation.

    The VES chi2 axis starts at VES_PLOT_MIN.
    """
    if fit_parameters is None:
        print(f'Skipping transmission plot for WP {working_point}: polynomial fit is unavailable.')
        return None
    ves_chi2 = np.sort(vesuvius_ves_df['chi2_cut'].dropna().unique().astype(float))
    ves_chi2 = ves_chi2[ves_chi2 >= VES_PLOT_MIN]
    fs_chi2 = np.polyval(fit_parameters, ves_chi2)
    transmissions = []
    transmission_errors = []
    valid_ves_chi2 = []
    for ves_cut, fs_cut in zip(ves_chi2, fs_chi2):
        ves_rate = interpolate_track_rate(vesuvius_ves_df, ves_cut)
        fs_rate = interpolate_track_rate(vesuvius_fs_df, fs_cut)
        ves_count = interpolate_track_rate(
            vesuvius_ves_df,
            ves_cut,
            column='n_tracks',
        )
        fs_count = interpolate_track_rate(
            vesuvius_fs_df,
            fs_cut,
            column='n_tracks',
        )
        if not np.isfinite(ves_rate) or not np.isfinite(fs_rate):
            continue
        if (
            fs_rate <= 0
            or not np.isfinite(ves_count)
            or not np.isfinite(fs_count)
            or ves_count <= 0
            or fs_count <= 0
        ):
            continue
        valid_ves_chi2.append(ves_cut)
        transmission = ves_rate / fs_rate
        ves_rate_error = interpolate_rate_error(vesuvius_ves_df, ves_cut)
        fs_rate_error = interpolate_rate_error(vesuvius_fs_df, fs_cut)
        statistical_variance = (ves_rate_error / fs_rate) ** 2 + (ves_rate * fs_rate_error / fs_rate ** 2) ** 2

        fit_cut_error = polynomial_prediction_sigma(
            ves_cut,
            fit_covariance,
            fit_degree,
        )
        cut_step = max(0.01, 0.001 * (vesuvius_fs_df['chi2_cut'].max() - vesuvius_fs_df['chi2_cut'].min()))
        lower_rate = interpolate_track_rate(vesuvius_fs_df, fs_cut - cut_step)
        upper_rate = interpolate_track_rate(vesuvius_fs_df, fs_cut + cut_step)
        if fit_cut_error > 0 and np.isfinite(lower_rate) and np.isfinite(upper_rate):
            rate_slope = (upper_rate - lower_rate) / (2.0 * cut_step)
            fit_variance = (ves_rate * rate_slope * fit_cut_error / fs_rate ** 2) ** 2
        else:
            fit_variance = 0.0

        transmissions.append(transmission)
        transmission_errors.append(np.sqrt(statistical_variance + fit_variance))
    if not transmissions:
        print(f'Skipping transmission plot for WP {working_point}: no valid fitted cut pairs.')
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        valid_ves_chi2,
        transmissions,
        yerr=transmission_errors,
        fmt='o-',
        markersize=3,
        linewidth=1.5,
        capsize=2,
        color='orchid',
        label='Transmission',
    )
    ax.set_xlabel(r'$\chi^2_{\rm Ves}$ cut')
    ax.set_ylabel('Transmission')
    ax.set_title(f'{detector} — WP {working_point}\nTransmission in Vesuvius region for $\\chi^2$-pairs satisfying $\\Phi_{{\\rm Ves}}^{{\\rm control}} / \\Phi_{{\\rm FS}}^{{\\rm control}} = 1$')
    ax.set_xlim(VES_PLOT_MIN, CHI2_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_file = output_dir / 'transmission_along_fit.png'
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
    return output_file


def plot_ratio_1figure(ratio_df, output_file='ratio_integer_FS_and_VES.svg', fs_cuts=None, detector=None, working_point=None, data_type=None, ves_cuts_to_plot=None, ratio_error_df=None, **kwargs):
    """
    Make one SVG figure with two orientations of the VES/FS ratio.

    LEFT:
        x-axis = VES chi2 cut
        curves = fixed FS chi2 cuts

    RIGHT:
        x-axis = FS chi2 cut
        curves = fixed VES chi2 cuts

    The x-axis starts at VES_PLOT_MIN.

    Extra keyword arguments are accepted for compatibility
    with the existing main() function.
    """
    ratio_df = ratio_df.copy()
    ratio_df.index = ratio_df.index.astype(float)
    ratio_df.columns = ratio_df.columns.astype(float)
    ratio_df = ratio_df.sort_index()
    ratio_df = ratio_df.reindex(sorted(ratio_df.columns), axis=1)
    available_fs_cuts = ratio_df.index.to_numpy(dtype=float)
    available_ves_cuts = ratio_df.columns.to_numpy(dtype=float)
    if fs_cuts is None:
        fs_cuts = [cut for cut in available_fs_cuts if np.isclose(cut, round(cut), atol=1e-08)]
    fs_cuts = np.asarray(fs_cuts, dtype=float)
    valid_fs_cuts = []
    for cut in fs_cuts:
        matches = np.where(np.isclose(available_fs_cuts, cut, atol=1e-08, rtol=0.0))[0]
        if len(matches) == 0:
            print(f'WARNING: FS cut {cut:g} not found; skipping.')
            continue
        valid_fs_cuts.append(available_fs_cuts[matches[0]])
    fs_cuts = np.asarray(valid_fs_cuts, dtype=float)
    if ves_cuts_to_plot is None:
        ves_cuts_to_plot = [cut for cut in available_ves_cuts if np.isclose(cut, round(cut), atol=1e-08) and cut >= VES_PLOT_MIN]
    ves_cuts_to_plot = np.asarray(ves_cuts_to_plot, dtype=float)
    valid_ves_cuts = []
    for cut in ves_cuts_to_plot:
        if cut < VES_PLOT_MIN:
            continue
        matches = np.where(np.isclose(available_ves_cuts, cut, atol=1e-08, rtol=0.0))[0]
        if len(matches) == 0:
            print(f'WARNING: VES cut {cut:g} not found; skipping.')
            continue
        valid_ves_cuts.append(available_ves_cuts[matches[0]])
    ves_cuts_to_plot = np.asarray(valid_ves_cuts, dtype=float)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, 13))

    for i, fs_cut in enumerate(fs_cuts):
        matches = np.where(np.isclose(available_fs_cuts, fs_cut, atol=1e-08, rtol=0.0))[0]
        if len(matches) == 0:
            continue
        row_idx = matches[0]
        ratios = ratio_df.iloc[row_idx].to_numpy(dtype=float)
        errors = None
        if ratio_error_df is not None:
            errors = ratio_error_df.iloc[row_idx].to_numpy(dtype=float)

        ves_mask = available_ves_cuts >= VES_PLOT_MIN

        ax_left.errorbar(
            available_ves_cuts[ves_mask],
            ratios[ves_mask],
            yerr=None if errors is None else errors[ves_mask],
            marker='o',
            fmt='o-',
            linewidth=1.5,
            markersize=3,
            color=colors[i],
            capsize=2,
            label=f'FS $\\chi^2$ < {fs_cut:.0f}'
        )
    ax_left.axhline(1.0, linestyle='--', linewidth=1.2, label='Ratio = 1')
    ax_left.set_xlabel(r'$\chi^2_{\rm Ves}$ cut')
    ax_left.set_ylabel(r'$\Phi_{\rm Ves}(\Delta\phi_{\rm control}, \Delta\theta_{\rm control}) / \Phi_{\rm FS}(\Delta\phi_{\rm control}, \Delta\theta_{\rm control})$')
    #ax_left.set_title('Fixed FS cuts')
    ax_left.set_xlim(VES_PLOT_MIN, CHI2_MAX)
    ax_left.set_xlim(VES_PLOT_MIN, CHI2_MAX)
    ax_left.set_ylim(0, 5)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=8, ncol=2)

    for i, ves_cut in enumerate(ves_cuts_to_plot):
        matches = np.where(np.isclose(available_ves_cuts, ves_cut, atol=1e-08, rtol=0.0))[0]
        if len(matches) == 0:
            continue
        col_idx = matches[0]
        ratios = ratio_df.iloc[:, col_idx].to_numpy(dtype=float)
        errors = None
        if ratio_error_df is not None:
            errors = ratio_error_df.iloc[:, col_idx].to_numpy(dtype=float)

        ax_right.errorbar(
            available_fs_cuts,
            ratios,
            yerr=errors,
            marker='o',
            fmt='o-',
            linewidth=1.5,
            markersize=3,
            color=colors[i],
            capsize=2,
            label=f'VES $\\chi^2$ < {ves_cut:.0f}'
        )
    ax_right.axhline(1.0, linestyle='--', linewidth=1.2, label='Ratio = 1')
    ax_right.set_xlabel('$\\chi^2_{{\\rm FS}}$ cut')
    #ax_right.set_title('Fixed VES cuts')
    ax_right.set_xlim(VES_PLOT_MIN, CHI2_MAX)
    ax_right.set_ylim(0, 5)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=8, ncol=2)
    title_parts = []
    if detector is not None:
        title_parts.insert(0, str(detector))
    if working_point is not None:
        title_parts.insert(1, f'WP {working_point}')
    fig.suptitle(' — '.join(title_parts), fontsize=14)
    fig.tight_layout()
    output_file = DEFAULT_OUTPUT_DIR / output_file
    fig.savefig(output_file, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved SVG: {output_file}')
    print('\n' + '=' * 80)
    print('NUMERICAL CHECK OF PLOTTED VALUES')
    print('=' * 80)
    diagnostic_ves_cuts = [1.0, 5.0, 10.0]
    for ves_cut in diagnostic_ves_cuts:
        matches = np.where(np.isclose(available_ves_cuts, ves_cut, atol=1e-08, rtol=0.0))[0]
        if len(matches) == 0:
            continue
        col_idx = matches[0]
        print(f'\nAt VES < {ves_cut:.0f}:')
        print('-' * 40)
        values = []
        for fs_cut in fs_cuts:
            fs_matches = np.where(np.isclose(available_fs_cuts, fs_cut, atol=1e-08, rtol=0.0))[0]
            if len(fs_matches) == 0:
                continue
            value = ratio_df.iloc[fs_matches[0], col_idx]
            values.append(value)
            print(f'  FS < {fs_cut:>2.0f}: {value:.6f}')
        finite_values = np.asarray(values)[np.isfinite(values)]
        if len(finite_values) >= 2:
            if np.all(np.diff(finite_values) <= 1e-12):
                print('  -> OK: ratio is non-increasing')
            else:
                print('  -> WARNING: ratio increases somewhere!')
    print('\n' + '=' * 80)
    print('MONOTONICITY CHECK — FIXED VES, VARYING FS')
    print('=' * 80)
    for ves_cut in ves_cuts_to_plot:
        matches = np.where(np.isclose(available_ves_cuts, ves_cut, atol=1e-08, rtol=0.0))[0]
        if len(matches) == 0:
            continue
        ratios = ratio_df.iloc[:, matches[0]].to_numpy(dtype=float)
        finite_ratios = ratios[np.isfinite(ratios)]
        if len(finite_ratios) < 2:
            print(f'VES < {ves_cut:.0f}: not enough finite points')
            continue
        differences = np.diff(finite_ratios)
        if np.all(differences <= 1e-12):
            print(f'VES < {ves_cut:>2.0f}: OK — non-increasing')
        else:
            print(f'VES < {ves_cut:>2.0f}: WARNING — increase detected')
    return output_file


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot VES/FS track-rate ratios from MURAVES chi2 scan parquet files.')
    parser.add_argument('--detector', type=str, default=DEFAULT_DETECTOR, help=f'Detector to process (default: {DEFAULT_DETECTOR}).')
    parser.add_argument('--working-points', type=int, nargs='+', default=None, help='Working points to plot. If omitted, all working points with both VES and FS parquet files are processed.')
    parser.add_argument('--fs-cuts', type=float, nargs='+', default=None, help='FS chi2 cuts for which to make plots. If omitted, all available FS chi2 cuts are plotted.')
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR, help=f'Directory containing scan parquet files (default: {DEFAULT_INPUT_DIR}).')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help=f'Base output directory (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--no-scan-csv', action='store_true', help='Do not save ratio_scan.csv.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    detector = args.detector
    input_dir = args.input_dir / detector
    vesuvius_input_dir = input_dir / 'vesuvius_region'
    output_base_dir = args.output_dir
    if args.working_points is None:
        working_points = find_common_working_points(input_dir)
        if not working_points:
            raise RuntimeError(f'No common VES/FS working points found in {input_dir}')
    else:
        working_points = sorted(set(args.working_points))
    print(f'Detector:        {detector}')
    print(f'Input directory: {input_dir}')
    print(f'Output directory: {output_base_dir}')
    print(f'Working points:  {working_points}')
    print(f'VES plot starts:  χ² = {VES_PLOT_MIN:g}')
    for working_point in working_points:
        print()
        print('=' * 70)
        print(f'Working point: {working_point}')
        print('=' * 70)
        try:
            ves_file, fs_file = get_parquet_files(input_dir, working_point)
        except FileNotFoundError as exc:
            print(f'WARNING: {exc}')
            print('Skipping this working point.')
            continue
        print(f'VES: {ves_file}')
        print(f'FS:  {fs_file}')
        ves_df = load_scan(ves_file)
        fs_df = load_scan(fs_file)
        ves_detectors = set(ves_df['detector'].dropna().astype(str))
        fs_detectors = set(fs_df['detector'].dropna().astype(str))
        if detector not in ves_detectors:
            print(f'WARNING: detector {detector} not found in VES file.')
        if detector not in fs_detectors:
            print(f'WARNING: detector {detector} not found in FS file.')
        ves_wps = set(ves_df['working_point'].dropna().astype(int))
        fs_wps = set(fs_df['working_point'].dropna().astype(int))
        if working_point not in ves_wps:
            print(f'WARNING: WP {working_point} not found in VES data.')
        if working_point not in fs_wps:
            print(f'WARNING: WP {working_point} not found in FS data.')
        ratio_df = calculate_ratio_table(ves_df, fs_df)
        ratio_error_df = calculate_ratio_error_table(ves_df, fs_df)
        available_fs_cuts = ratio_df.index.to_numpy(dtype=float)
        if args.fs_cuts is None:
            fs_cuts = available_fs_cuts
        else:
            fs_cuts = select_requested_cuts(available_fs_cuts, args.fs_cuts)
        wp_output_dir = output_base_dir / detector / f'WP{working_point}'
        wp_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'Creating {len(fs_cuts)} plot(s)...')
        contour_output_file, fit_parameters, fit_covariance = plot_ratio_one_contour(ratio_df=ratio_df, detector=detector, working_point=working_point, output_dir=wp_output_dir, tolerance=0.005)
        try:
            vesuvius_ves_file, vesuvius_fs_file = get_parquet_files(vesuvius_input_dir, working_point)
        except FileNotFoundError as exc:
            print(f'WARNING: {exc}')
            print('Skipping transmission plot for this working point.')
        else:
            vesuvius_ves_df = load_scan(vesuvius_ves_file)
            vesuvius_fs_df = load_scan(vesuvius_fs_file)
            transmission_output_file = plot_transmission_along_fit(vesuvius_ves_df=vesuvius_ves_df, vesuvius_fs_df=vesuvius_fs_df, fit_parameters=fit_parameters, fit_covariance=fit_covariance, detector=detector, working_point=working_point, output_dir=wp_output_dir)
            if transmission_output_file is not None:
                print(f'  {transmission_output_file}')
        available_fs_cuts = ratio_df.index.to_numpy(dtype=float)
        if args.fs_cuts is None:
            fs_cuts = available_fs_cuts[np.isclose(available_fs_cuts, np.round(available_fs_cuts), atol=1e-06, rtol=0)]
        else:
            fs_cuts = select_requested_cuts(available_fs_cuts, args.fs_cuts)
        print(f'Creating one plot with {len(fs_cuts)} integer FS chi2 cuts...')
        output_file = plot_ratio_1figure(ratio_df=ratio_df, ratio_error_df=ratio_error_df, fs_cuts=fs_cuts, detector=detector, working_point=working_point, output_dir=wp_output_dir)
        print(f'  {output_file}')
        print(f'Saved ratio = 1 plot: {output_file}')
    print()
    print('Done.')
if __name__ == '__main__':
    main()