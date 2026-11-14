"""WavLM speaker segmentation model with a `diarize()` method for complete recordings.

    from transformers import AutoModel
    model = AutoModel.from_pretrained("<repo>", trust_remote_code=True)
    for turn, _, speaker in model.diarize("meeting.wav").itertracks(yield_label=True):
        print(f"{turn.start:7.1f} {turn.end:7.1f}  {speaker}")

The network itself labels the speakers of one 10-second chunk. `diarize()` puts it into the
pyannote.audio pipeline, which slides it over the recording, computes a voice embedding for
every local speaker and clusters them into the speakers of the whole recording.
"""

import torch
from torch import nn
from transformers import PreTrainedModel, WavLMConfig, WavLMModel
from transformers.modeling_outputs import TokenClassifierOutput

from .configuration_wavlm_diarization import WavLMDiarizationConfig

SAMPLE_RATE = 16000
FRAME_STEP = 320  # one output frame every 20 ms ...
FRAME_SIZE = 400  # ... each looking at 25 ms of audio


class Head(nn.Module):
    """Features (batch, frames, dim) -> logits (batch, frames, classes)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=layers, bidirectional=True, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return self.out(x)


class WavLMDiarizationModel(PreTrainedModel):
    config_class = WavLMDiarizationConfig
    main_input_name = "input_values"

    def __init__(self, config: WavLMDiarizationConfig):
        super().__init__(config)
        wavlm_config = WavLMConfig(**config.wavlm_config)
        self.wavlm = WavLMModel(wavlm_config)
        self.layer_weights = nn.Parameter(torch.zeros(wavlm_config.num_hidden_layers + 1))
        self.head = Head(wavlm_config.hidden_size, config.hidden, config.layers, config.dropout, config.num_classes)
        self._pipeline = None
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        pass  # all weights come from the checkpoint

    def forward(self, input_values: torch.Tensor, **kwargs) -> TokenClassifierOutput:
        """Waveforms (batch, samples) at 16 kHz -> logits (batch, frames, 7), one frame every 20 ms.

        The 7 classes are: nobody, speaker 1, 2, 3, speakers 1+2, 1+3, 2+3.
        """
        layers = self.wavlm(input_values, output_hidden_states=True).hidden_states
        weights = torch.softmax(self.layer_weights, dim=0)
        features = sum(w * layer for w, layer in zip(weights, layers))
        return TokenClassifierOutput(logits=self.head(features))

    # ---- diarization of a complete recording (needs `pip install pyannote.audio`)

    def _build_pipeline(self):
        from pyannote.audio import Model
        from pyannote.audio.core.plda import PLDA
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from pyannote.audio.pipelines import SpeakerDiarization

        net, config = self, self.config

        class Segmentation(Model):
            """Makes this network look like a pyannote segmentation model."""

            def __init__(self):
                super().__init__(sample_rate=SAMPLE_RATE, num_channels=1)
                self.net = net
                self.specifications = Specifications(
                    problem=Problem.MONO_LABEL_CLASSIFICATION,
                    resolution=Resolution.FRAME,
                    duration=config.chunk_sec,
                    classes=[f"speaker#{i + 1}" for i in range(config.max_speakers)],
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
                return torch.log_softmax(self.net(waveforms.squeeze(1)).logits.float(), dim=-1)

        pipeline = SpeakerDiarization(
            segmentation=Segmentation(),
            embedding=config.embedding,
            embedding_exclude_overlap=True,
            clustering="AgglomerativeClustering",
            # PLDA is only used by pyannote's VBx clustering, not here. An empty
            # placeholder stops pyannote from downloading it from a gated repository.
            plda=PLDA.__new__(PLDA),
            segmentation_batch_size=16,
            embedding_batch_size=16,
            legacy=True,  # return a plain Annotation
        )
        pipeline.instantiate({
            "segmentation": {"min_duration_off": 0.0},
            "clustering": {
                "method": "centroid",
                "min_cluster_size": config.min_cluster_size,
                "threshold": config.clustering_threshold,
            },
        })
        return pipeline

    @torch.no_grad()
    def diarize(
        self,
        audio,
        sample_rate: int | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        """Who spoke when in a recording of any length.

        audio: path to an audio file, or a waveform tensor (samples,) or (channels, samples)
               together with its `sample_rate`.
        The number of speakers is found automatically unless `num_speakers` (or a range) is given.
        Returns a `pyannote.core.Annotation`; iterate over it with
        `for turn, _, speaker in annotation.itertracks(yield_label=True)`.
        """
        self.eval()
        device = self.device  # remembered now: building the pipeline moves the model to the CPU
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        self._pipeline.to(device)

        if isinstance(audio, torch.Tensor):
            if sample_rate is None:
                raise ValueError("Give the sample_rate of the waveform.")
            waveform = audio.float().reshape(1, -1) if audio.ndim == 1 else audio.float()
        else:
            import soundfile as sf

            wav, sample_rate = sf.read(str(audio), dtype="float32", always_2d=True)  # (samples, channels)
            waveform = torch.from_numpy(wav.T)
        return self._pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers,
        )
