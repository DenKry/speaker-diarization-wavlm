"""Diarization of a whole recording.

Our model only labels speakers inside one 10-second chunk. To diarize a full
meeting we put it into the pyannote pipeline, which slides the model over the
recording, computes a voice embedding for every local speaker and clusters
them into global speakers.

The pipeline is configured like `pyannote/speaker-diarization-3.1`. With
`checkpoint=None` it uses pyannote's own segmentation model (our baseline B1),
so the baseline and our models differ only in the segmentation model.
"""

import soundfile as sf
import torch
from pyannote.audio import Model
from pyannote.audio.core.plda import PLDA
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Annotation

from diar.data.dataset import FRAME_SIZE, FRAME_STEP, MAX_SPEAKERS
from diar.model import build_model

# Hyper-parameters of pyannote/speaker-diarization-3.1
PIPELINE_PARAMS = {
    "segmentation": {"min_duration_off": 0.0},
    "clustering": {"method": "centroid", "min_cluster_size": 12, "threshold": 0.7045654963945799},
}


class PyannoteSegmentation(Model):
    """Makes our network look like a pyannote segmentation model."""

    def __init__(self, net: torch.nn.Module, chunk_sec: float = 10.0):
        super().__init__(sample_rate=16000, num_channels=1)
        self.net = net
        self.specifications = Specifications(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=chunk_sec,
            classes=[f"speaker#{i + 1}" for i in range(MAX_SPEAKERS)],
            powerset_max_classes=2,
            permutation_invariant=True,
        )

    # pyannote uses these three methods to know which time span each output frame covers
    def num_frames(self, num_samples: int) -> int:
        return (num_samples - FRAME_SIZE) // FRAME_STEP + 1

    def receptive_field_size(self, num_frames: int = 1) -> int:
        return FRAME_SIZE + (num_frames - 1) * FRAME_STEP

    def receptive_field_center(self, frame: int = 0) -> int:
        return frame * FRAME_STEP + FRAME_SIZE // 2

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """(batch, 1, samples) -> log-probabilities (batch, frames, 7)."""
        return torch.log_softmax(self.net(waveforms.squeeze(1)).float(), dim=-1)


def load_model(checkpoint: str) -> torch.nn.Module:
    """Rebuild a trained model from a checkpoint written by the trainer."""
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.eval()


def build_pipeline(checkpoint: str | None = None) -> SpeakerDiarization:
    """Full pipeline with our model, or with pyannote's own one if checkpoint is None."""
    if checkpoint is None:
        segmentation = "pyannote/segmentation-3.0"
    else:
        segmentation = PyannoteSegmentation(load_model(checkpoint))
    pipeline = SpeakerDiarization(
        segmentation=segmentation,
        embedding="pyannote/wespeaker-voxceleb-resnet34-LM",
        embedding_exclude_overlap=True,
        clustering="AgglomerativeClustering",
        # PLDA is only used by pyannote's newer VBx clustering, not by ours. An empty
        # placeholder stops pyannote from downloading it from a gated repository.
        plda=PLDA.__new__(PLDA),
        segmentation_batch_size=16,
        embedding_batch_size=16,
        legacy=True,  # return a plain Annotation
    )
    pipeline.instantiate(PIPELINE_PARAMS)
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return pipeline


def diarize(pipeline: SpeakerDiarization, audio_path: str) -> Annotation:
    """Who spoke when in an audio file."""
    wav, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)  # (samples, channels)
    return pipeline({"waveform": torch.from_numpy(wav.T), "sample_rate": sample_rate})
