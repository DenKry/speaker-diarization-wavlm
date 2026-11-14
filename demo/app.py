"""Live demo:  python demo/app.py --checkpoint runs/e3_finetune/best.pt [--baseline]

Upload or record audio and see who spoke when.
"""

import argparse

import gradio as gr
from pyannote.metrics.diarization import DiarizationErrorRate

from diar.eval.plots import plot_timeline
from diar.inference.pipeline import build_pipeline, diarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", action="store_true", help="also show pyannote for comparison")
    parser.add_argument("--share", action="store_true", help="public link (needed on Colab)")
    args = parser.parse_args()

    pipelines = {"Our model": build_pipeline(args.checkpoint)}
    if args.baseline:
        pipelines["pyannote"] = build_pipeline(None)

    def run(audio_path: str):
        if audio_path is None:
            return None, "Upload or record some audio first."
        results = {name: diarize(p, audio_path) for name, p in pipelines.items()}
        ours = results["Our model"]
        if "pyannote" in results:
            # Speaker names are arbitrary in each system: rename pyannote's to match ours.
            mapping = DiarizationErrorRate().optimal_mapping(ours, results["pyannote"])
            results["pyannote"] = results["pyannote"].rename_labels(mapping)
        turns = [f"{s.start:7.1f}s - {s.end:7.1f}s  {speaker}" for s, _, speaker in ours.itertracks(yield_label=True)]
        return plot_timeline(results), f"{len(ours.labels())} speakers found\n\n" + "\n".join(turns)

    gr.Interface(
        fn=run,
        inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio"),
        outputs=[gr.Plot(label="Timeline"), gr.Textbox(label="Speech turns (our model)", lines=12)],
        title="Speaker diarization with WavLM",
        flagging_mode="never",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
