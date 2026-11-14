"""Diarization Error Rate and Jaccard Error Rate on a prepared AMI split."""

import csv
from pathlib import Path

import soundfile as sf
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from diar.data.dataset import load_rttm
from diar.inference.pipeline import diarize

COMPONENTS = {"missed": "missed detection", "false_alarm": "false alarm", "confusion": "confusion"}


def new_metrics() -> dict:
    """DER in two settings:
      strict - every frame counts, overlapped speech included;
      collar - 0.25 s around each reference boundary is not scored
               (pyannote's `collar` is the total width, hence 0.5).
    JER (strict): the error is computed per speaker and averaged over speakers,
    so a person who talks little weighs as much as one who talks a lot.
    """
    return {
        "strict": DiarizationErrorRate(collar=0.0, skip_overlap=False),
        "collar": DiarizationErrorRate(collar=0.5, skip_overlap=False),
        "jer": JaccardErrorRate(collar=0.0, skip_overlap=False),
    }


def read_annotation(rttm: Path) -> Annotation:
    annotation = Annotation(uri=rttm.stem)
    for i, (start, end, speaker) in enumerate(load_rttm(rttm)):
        annotation[Segment(start, end), i] = speaker
    return annotation


def score_meeting(metrics: dict, reference: Annotation, hypothesis: Annotation, audio: Path) -> None:
    whole_file = Timeline([Segment(0.0, sf.info(str(audio)).duration)])
    v = {k: m(reference, hypothesis, uem=whole_file) for k, m in metrics.items()}
    print(f"{audio.stem}: DER {v['strict']:.2%} (strict), {v['collar']:.2%} (collar) | JER {v['jer']:.2%}", flush=True)


def save_row(metrics: dict, name: str, split: str, results_csv: str) -> dict:
    """Add the scores accumulated over all meetings to the results table."""
    row = {"experiment": name, "split": split}
    for setting in ("strict", "collar"):
        metric = metrics[setting]
        row[f"der_{setting}"] = round(100 * abs(metric), 2)
        for short, key in COMPONENTS.items():
            row[f"{short}_{setting}"] = round(100 * metric[key] / metric["total"], 2)
    row["jer"] = round(100 * abs(metrics["jer"]), 2)

    results = Path(results_csv)
    results.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results.exists()
    with open(results, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n{name}: " + ", ".join(f"{k} {v}" for k, v in row.items() if k not in ("experiment", "split")))
    return row


def evaluate(pipeline, split_dir: str, out_dir: str, name: str, results_csv: str = "results/results.csv",
             resume: bool = False) -> dict:
    """Diarize every meeting, save the RTTMs and add the scores to the results table.

    resume=True reuses the RTTMs that an interrupted run has already written to out_dir.
    """
    metrics = new_metrics()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for rttm in sorted(Path(split_dir).glob("*.rttm")):
        audio = rttm.with_suffix(".flac")
        if resume and (out / rttm.name).exists():
            hypothesis = read_annotation(out / rttm.name)
        else:
            hypothesis = diarize(pipeline, str(audio))
            hypothesis.uri = rttm.stem
            with open(out / rttm.name, "w") as f:
                hypothesis.write_rttm(f)
        score_meeting(metrics, read_annotation(rttm), hypothesis, audio)

    return save_row(metrics, name, Path(split_dir).name, results_csv)


def evaluate_rttms(rttm_dir: str, split_dir: str, name: str, results_csv: str = "results/results.csv",
                   exclude: tuple = ()) -> dict:
    """Score RTTMs that some other system has already written (one per meeting)."""
    metrics = new_metrics()
    for rttm in sorted(Path(split_dir).glob("*.rttm")):
        if rttm.stem in exclude:  # a recording left out of the total, e.g. an outlier
            continue
        hypothesis = read_annotation(Path(rttm_dir) / rttm.name)
        score_meeting(metrics, read_annotation(rttm), hypothesis, rttm.with_suffix(".flac"))
    return save_row(metrics, name, Path(split_dir).name, results_csv)
