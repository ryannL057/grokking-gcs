import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def reducers():
    try:
        return importlib.import_module('gcs.reproduce.statistics')
    except ModuleNotFoundError:
        pytest.fail('Cached statistical reproduction is not implemented')


def test_primary_counts_include_all_depths_and_cross_path_controls():
    result = reducers().primary(ROOT / 'results/reference')
    assert result['n'] == 104
    assert result['attention_rise_dip'] == 94
    assert result['mlp_fall_recovery'] == 103
    assert result['attention_fall_recovery'] == 98
    assert result['mlp_rise_dip'] == 60
    assert result['joint'] == 93
    assert result['by_depth']['2']['n'] == 34


def test_freeze_counts_are_computed_from_153_raw_records():
    result = reducers().freeze(ROOT / 'results/reference')
    assert result['records'] == 153
    assert result['successes'] == dict(VV=3, VL=15, LV=15, LL=16, **{'0L':17,'0V':4,'0V_long':8,'V_unfrozen':12,'L_attn_frozen':17})
    assert result['budget_range'] == [2000,14800]


def test_branch_success_after_matched_budget_is_censored():
    result = reducers().branch_outcome({'branch_epoch': 100, 'curve': {'epochs': [100,150,250], 'test_acc': [.4,.8,1.]}}, 100)
    assert result == {'success': False, 'first_success_updates': None}


def test_reset_mean_is_recomputed_from_paired_runs():
    result = reducers().resets(ROOT / 'results/reference')
    assert result['n'] == 17
    assert result['attention_hurts_more'] == 17
    assert result['mean_attention_drop'] == pytest.approx(.3979718747326652)
    assert result['mean_mlp_drop'] == pytest.approx(.031191803267900733)


def test_matched_null_uses_max_within_each_run_then_mean():
    result = reducers().submodules(ROOT / 'results/reference')
    assert len(result['matched_per_run']) == 17
    assert result['matched_mean_of_run_maxima'] == pytest.approx(.025,abs=.0005)


def test_current_random_window_table_and_fourier_trend_controls():
    result = reducers().workflow(ROOT / 'results/reference')
    row=next(r for r in result['summary'] if r['selector']=='random' and r['target']=='attention_equivariance_drop')
    assert row['n_runs'] == 11
    assert row['median_percentile'] == pytest.approx(.511,abs=.0005)
    assert reducers().fourier(ROOT / 'results/reference')['medians']['rho_differences'] == pytest.approx(.034210526315789476)


def test_table_six_exports_missing_historical_metadata_as_json_null(tmp_path):
    import json
    from gcs.reproduce.cli import main
    assert main(ROOT,['--target','tab06','--output',str(tmp_path/'table')]) == 0
    data=json.loads((tmp_path/'table/tab06.json').read_text())
    assert len(data['per_run']) == 104


def test_plotting_ignores_external_interactive_backend(tmp_path,monkeypatch):
    monkeypatch.setenv('MPLBACKEND','kitcat')
    from gcs.reproduce.render import render_neighbors
    paths=render_neighbors(ROOT/'results/reference',tmp_path)
    assert (tmp_path/'fig08.pdf').stat().st_size>1000
    assert 'fig08.svg' in paths


def test_missing_original_cannot_report_success_or_emit_a_fake_figure(tmp_path):
    import json
    from gcs.reproduce.cli import main
    out=tmp_path/'missing'
    assert main(ROOT,['--target','gap_jvp_norms','--output',str(out)])==1
    report=json.loads((out/'report.json').read_text())
    assert report['complete'] is False
    assert report['targets'][0]['status']=='incomplete'
    assert not list(out.glob('*.png'))


def ablations():
    try:
        return importlib.import_module('gcs.reproduce.ablations')
    except ModuleNotFoundError:
        pytest.fail('Recovered ablation reproduction is not implemented')


def test_recovered_figures_render_from_verified_numerical_inputs(tmp_path):
    import json
    from gcs.reproduce.cli import main
    out = tmp_path / 'ablations'
    assert main(ROOT, ['--target', 'fig06', '--target', 'fig07', '--output', str(out)]) == 0
    report = json.loads((out / 'report.json').read_text())
    assert report['complete'] is True
    assert [row['status'] for row in report['targets']] == ['passed', 'passed']
    for target in ['fig06', 'fig07']:
        for suffix in ['pdf', 'png', 'svg']:
            assert (out / f'{target}.{suffix}').stat().st_size > 1000


def test_recovered_ablations_preserve_raw_precision_and_unequal_horizons():
    import json
    ref = ROOT / 'results/reference'
    d = ablations().load_trajectories(ref, 'd')
    n = ablations().load_trajectories(ref, 'N')
    assert list(d) == [2, 4, 6, 10, 12, 14]
    assert {key: len(row['epochs']) for key, row in d.items()} == dict.fromkeys(d, 75)
    assert {key: (len(row['epochs']), row['epochs'][-1]) for key, row in n.items()} == {
        100: (69, 13600), 200: (76, 14999), 400: (69, 13600),
        750: (69, 13600), 1850: (75, 14800),
    }
    for parameter, result in [('d', d), ('n', n)]:
        source = json.loads((ref / f'evidence/ablations/{parameter}_trajectories.json').read_text())
        assert {str(key): row for key, row in result.items()} == source


@pytest.mark.parametrize('problem', ['length', 'missing_group', 'empty', 'unordered', 'nonfinite'])
def test_ablation_loader_rejects_incomplete_or_invalid_curves(tmp_path, problem):
    import json
    data = {str(key): dict(epochs=[0, 200], overall_gcs=[1., 2.],
                          attention_gcs=[2., 3.], mlp_gcs=[3., 4.])
            for key in [2, 4, 6, 10, 12, 14]}
    if problem == 'length':
        data['2']['attention_gcs'].pop()
    elif problem == 'missing_group':
        del data['14']
    elif problem == 'empty':
        data['2'] = {key: [] for key in data['2']}
    elif problem == 'unordered':
        data['2']['epochs'] = [200, 0]
    else:
        data['2']['mlp_gcs'][0] = float('nan')
    folder = tmp_path / 'evidence/ablations'
    folder.mkdir(parents=True)
    (folder / 'd_trajectories.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ablations().load_trajectories(tmp_path, 'd')


def test_ablation_plot_scales_each_curve_and_keeps_each_epoch_grid(monkeypatch):
    import numpy as np
    monkeypatch.setenv('MPLBACKEND', 'kitcat')
    data = {
        100: dict(epochs=[0, 5, 8], overall_gcs=[-2., 4., 2.],
                  attention_gcs=[-4., -2., 4.], mlp_gcs=[4., 1., 0.]),
        200: dict(epochs=[0, 7], overall_gcs=[3., 3.],
                  attention_gcs=[2., 6.], mlp_gcs=[6., 2.]),
    }
    fig = ablations().plot_trajectories(data, 'N')
    import matplotlib.pyplot as plt
    try:
        expected = [([0., 1., 2/3], [0., 0.]),
                    ([0., .25, 1.], [0., 1.]),
                    ([1., .25, 0.], [1., 0.])]
        assert len(fig.axes) == 3
        for axis, (first, second) in zip(fig.axes, expected):
            assert len(axis.lines) == 2
            np.testing.assert_array_equal(axis.lines[0].get_xdata(), [0, 5, 8])
            np.testing.assert_array_equal(axis.lines[1].get_xdata(), [0, 7])
            np.testing.assert_allclose(axis.lines[0].get_ydata(), first)
            np.testing.assert_allclose(axis.lines[1].get_ydata(), second)
    finally:
        plt.close(fig)
