# Copy from fairseq https://github.com/facebookresearch/fairseq

from dataclasses import _MISSING_TYPE,  dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class BaseConfig:
    """base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")

    @classmethod
    def from_namespace(cls, args):
        if isinstance(args, cls):
            return args
        else:
            config = cls()
            for k in config.__dataclass_fields__.keys():
                if k.startswith("_"):
                    # private member, skip
                    continue
                if hasattr(args, k):
                    setattr(config, k, getattr(args, k))

            return config


@dataclass
class OptimizationConfig(BaseConfig):
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
    )
    lr: float = field(
        default=2e-4,
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    pretrained_lr: float = field(
        default=1e-5,
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    missing_lr: float = field(
        default=2e-4,
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    freeze_pretrained: bool = field(
        default=False,
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    stop_min_lr: float = field(
        default=-1.0,
        metadata={
            "help": "stop training when the learning rate reaches this minimum"},
    )
    skip_remainder_batch: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, include the last (partial) batch of each epoch in training"
            " (default is to skip it)."
        },
    )
    amsgrad: bool = field(
        default=True,
        metadata={
            "help": "specify global optimizer for syncing models on different GPUs/shards"
        },
    )
    weight_decay: float = field(
        default=3e-9,
        metadata={
            "help": "weight decay (L2 penalty)"
        },
    )
    use_lr_scheduler: bool = field(
        default=True,
        metadata={
            "help": "whether to use lr_scheduler"
        },
    )
    num_warmup_steps: int = field(
        default=10000,
        metadata={
            "help": "specify global optimizer for syncing models on different GPUs/shards"
        },
    )


@dataclass
class CheckpointConfig(BaseConfig):
    save_dir: str = field(
        default="checkpoints", metadata={"help": "path to save checkpoints"}
    )
    restore_file: str = field(
        default="checkpoint_last.pt",
        metadata={
            "help": "filename from which to load checkpoint "
            "(default: <save-dir>/checkpoint_last.pt"
        },
    )
    pretrained_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "finetune from a pretrained model; note that meters and lr scheduler will be reset"
        },
    )
    reset_dataloader: bool = field(
        default=False,
        metadata={
            "help": "if set, does not reload dataloader state from the checkpoint"
        },
    )
    reset_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": "if set, does not load lr scheduler state from the checkpoint"
        },
    )
    reset_meters: bool = field(
        default=False,
        metadata={"help": "if set, does not load meters from the checkpoint"},
    )
    reset_optimizer: bool = field(
        default=False,
        metadata={
            "help": "if set, does not load optimizer state from the checkpoint"},
    )
    save_interval: int = field(
        default=10, metadata={"help": "save a checkpoint every N epochs"}
    )
    no_save: bool = field(
        default=False, metadata={"help": "don't save models or checkpoints"}
    )
    keep_epoch_checkpoints: bool = field(
        default=False, metadata={"help": "only store last and best checkpoints"}
    )
    no_last_checkpoints: bool = field(
        default=False, metadata={"help": "don't store last checkpoints"}
    )
    no_best_checkpoints: int = field(
        default=False, metadata={"help": "keep best N checkpoints based on scores"}
    )
    no_save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "don't save optimizer-state as part of checkpoint"},
    )
    best_checkpoint_metric: str = field(
        default="loss", metadata={"help": 'metric to use for saving "best" checkpoints'}
    )
    maximize_best_checkpoint_metric: bool = field(
        default=True,
        metadata={
            "help": 'select the largest metric value for saving "best" checkpoints'
        },
    )
    patience: int = field(
        default=-1,
        metadata={
            "help": (
                "early stop training if valid performance doesn't "
                "improve for N consecutive validation runs; note "
                "that this is influenced by --validate-interval"
            )
        },
    )


@dataclass
class GenerationConfig(BaseConfig):
    beam: int = field(
        default=2,
        metadata={"help": "beam size"},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "number of hypotheses to output"},
    )
    max_len_a: float = field(
        default=0,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    max_len_b: int = field(
        default=200,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    min_len: int = field(
        default=1,
        metadata={"help": "minimum generation length"},
    )
    match_source_len: bool = field(
        default=False,
        metadata={"help": "generations should match the source length"},
    )
    no_early_stop: bool = field(
        default=False,
        metadata={"help": "deprecated"},
    )
    no_beamable_mm: bool = field(
        default=False,
        metadata={"help": "don't use BeamableMM in attention layers"},
    )
    lenpen: float = field(
        default=1,
        metadata={
            "help": "length penalty: <1.0 favors shorter, >1.0 favors longer sentences"
        },
    )
    unkpen: float = field(
        default=0,
        metadata={
            "help": "unknown word penalty: <0 produces more unks, >0 produces fewer"
        },
    )
    replace_unk: Optional[str] = field(
        default=None,
        metadata={
            "help": "perform unknown replacement (optionally with alignment dictionary)",
            "argparse_const": "@@ ",
        },
    )
    sacrebleu: bool = field(
        default=False,
        metadata={"help": "score with sacrebleu"},
    )
    score_reference: bool = field(
        default=False,
        metadata={"help": "just score the reference translation"},
    )
    prefix_size: int = field(
        default=0,
        metadata={"help": "initialize generation by target prefix of given length"},
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": "ngram blocking such that this size ngram cannot be repeated in the generation"
        },
    )
    sampling: bool = field(
        default=False,
        metadata={"help": "sample hypotheses instead of using beam search"},
    )
    sampling_topk: int = field(
        default=-1,
        metadata={
            "help": "sample from top K likely next words instead of all words"},
    )
    sampling_topp: float = field(
        default=-1.0,
        metadata={
            "help": "sample from the smallest set whose cumulative probability mass exceeds p for next words"
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for generation"},
    )
    diverse_beam_groups: int = field(
        default=-1,
        metadata={"help": "number of groups for Diverse Beam Search"},
    )
    diverse_beam_strength: float = field(
        default=0.5,
        metadata={"help": "strength of diversity penalty for Diverse Beam Search"},
    )
    diversity_rate: float = field(
        default=-1.0,
        metadata={
            "help": "strength of diversity penalty for Diverse Siblings Search"},
    )
    print_step: bool = field(
        default=False,
        metadata={"help": "print steps"},
    )
    lm_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    lm_weight: float = field(
        default=0.0,
        metadata={"help": "weight for lm probs for lm fusion"},
    )

    # arguments for iterative refinement generator
    iter_decode_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "if > 0.0, it penalized early-stopping in decoding."},
    )
    iter_decode_max_iter: int = field(
        default=10,
        metadata={"help": "maximum iterations for iterative refinement."},
    )
    iter_decode_force_max_iter: bool = field(
        default=False,
        metadata={
            "help": "if set, run exact the maximum number of iterations without early stop"
        },
    )
    iter_decode_with_beam: int = field(
        default=1,
        metadata={
            "help": "if > 1, model will generate translations varying by the lengths."
        },
    )
    iter_decode_with_external_reranker: bool = field(
        default=False,
        metadata={
            "help": "if set, the last checkpoint are assumed to be a reranker to rescore the translations"
        },
    )
    retain_iter_history: bool = field(
        default=False,
        metadata={
            "help": "if set, decoding returns the whole history of iterative refinement"
        },
    )
    retain_dropout: bool = field(
        default=False,
        metadata={"help": "Use dropout at inference time"},
    )
    # temporarily set to Any until https://github.com/facebookresearch/hydra/issues/1117 is fixed
    # retain_dropout_modules: Optional[List[str]] = field(
    retain_dropout_modules: Any = field(
        default=None,
        metadata={
            "help": "if set, only retain dropout for the specified modules; "
            "if not set, then dropout will be retained for all modules"
        },
    )
    no_seed_provided: bool = field(
        default=False,
        metadata={
            "help": "if set, dont use seed for initializing random generators"},
    )
