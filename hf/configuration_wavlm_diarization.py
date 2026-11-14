"""Configuration of the WavLM speaker segmentation model."""

from transformers import PretrainedConfig, WavLMConfig


class WavLMDiarizationConfig(PretrainedConfig):
    model_type = "wavlm-diarization"

    def __init__(
        self,
        wavlm_config: dict | None = None,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 7,
        max_speakers: int = 3,
        chunk_sec: float = 10.0,
        embedding: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        clustering_threshold: float = 0.7045654963945799,
        min_cluster_size: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # settings of the WavLM encoder (no masking, no layer dropping: plain features)
        self.wavlm_config = wavlm_config or WavLMConfig(mask_time_prob=0.0, layerdrop=0.0).to_dict()
        # BiLSTM head
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        # 7 powerset classes: nobody, 3 single speakers, 3 pairs
        self.num_classes = num_classes
        self.max_speakers = max_speakers
        self.chunk_sec = chunk_sec
        # second stage: speaker embeddings + clustering (settings of pyannote/speaker-diarization-3.1)
        self.embedding = embedding
        self.clustering_threshold = clustering_threshold
        self.min_cluster_size = min_cluster_size
