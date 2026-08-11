"""Fit the single Zipfian baseline slope the simulation applies to every word.

The simulation shifts each corpus's sense-distribution slope by an offset from a
baseline, and the analysis grids results by that offset. For an offset to mean the
same thing in every cell -- across lemmata *and* across parts of speech -- the
baseline has to be one number for the whole vocabulary, not a per-word fit. See
:func:`simulation.zipfian.estimate_pooled_slope` for the statistical argument.

That makes the baseline a run-wide constant shared by all four per-PoS simulation
runs, so it is fitted once here and passed to each run via ``simulate_data.py
--baseline-slope``. Fitting it inside each run would defeat the purpose: each PoS
would get its own baseline and the cross-PoS comparison would stay broken.

Pass every vocab file that takes part in the comparison::

    python fit_baseline_slope.py source_data/word_sense_disambigation_corpora \\
        source_data/vocabs/most_diverse_{noun,verb,adj,adv}.json

It prints the slope and writes a small JSON report next to the first vocab file (or
wherever ``--report`` points) recording which vocabs and how many words went into it,
so a simulation run's baseline can be traced back to its provenance.
"""

import json
from pathlib import Path

import click

from data_processing.lwsd_loading import load_wsd
from simulation.zipfian import estimate_pooled_slope, sense_counts_for_words


def _load_targets(path: Path) -> list[tuple[str, str]]:
    """Read a JSON list of [lemma, pos] pairs into a list of tuples."""
    pairs = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(pair) for pair in pairs]


@click.command()
@click.argument("wsd_dir", type=Path)
@click.argument("targets_fps", type=Path, nargs=-1, required=True)
@click.option(
    "--report",
    type=Path,
    default=None,
    help="Where to write the provenance JSON (default: alongside the first vocab).",
)
def fit_baseline(wsd_dir: Path, targets_fps: tuple[Path, ...], report: Path | None) -> None:
    """Fit one pooled Zipf slope over every word in TARGETS_FPS."""
    wsd_df = load_wsd([wsd_dir])

    # De-duplicate across vocab files: a lemma appearing under two PoS is two distinct
    # targets, but the same (lemma, pos) listed twice must not be counted twice.
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for fp in targets_fps:
        for target in _load_targets(fp):
            if target not in seen:
                seen.add(target)
                targets.append(target)

    counts = sense_counts_for_words(wsd_df, targets)
    fit = estimate_pooled_slope(counts)

    if fit.status != "ok":
        raise click.ClickException(
            f"Pooled slope not fittable (status={fit.status}) over "
            f"{len(targets)} target(s) from {len(targets_fps)} vocab file(s)."
        )

    click.echo(
        f"pooled baseline slope = {fit.slope:.6f}  (SE {fit.se:.6f}; "
        f"{fit.n_words} words, {fit.n_obs} sense occurrences)"
    )
    click.echo(f"\nPass to the simulation as:  --baseline-slope {fit.slope:.6f}")

    report_path = report or targets_fps[0].parent / "baseline_slope.json"
    report_path.write_text(
        json.dumps(
            {
                "baseline_slope": fit.slope,
                "se": fit.se,
                "n_words": fit.n_words,
                "n_obs": fit.n_obs,
                "status": fit.status,
                "wsd_dir": str(wsd_dir),
                "vocabs": [str(fp) for fp in targets_fps],
                "n_targets_requested": len(targets),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    click.echo(f"Wrote {report_path}")


if __name__ == "__main__":
    fit_baseline()
