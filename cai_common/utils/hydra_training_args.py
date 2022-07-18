from dataclasses import dataclass, fields
from transformers import TrainingArguments, Seq2SeqTrainingArguments
from transformers.trainer_utils import IntervalStrategy, SchedulerType, HubStrategy
from transformers.training_args import OptimizerNames


class HydraToHFConverterMixIn:
    _hf_base_class = None


    @classmethod
    def as_hf_training_args(cls, cfg):
        """Converts a Hydra parsed config to a dictionary that can be used to instantiate a Hugging Face base class,
        specified in the static member _hf_base_class.

        This is needed because of a bug in Hydra: it doesn't ignore fields with init=False, such as _n_gpu.
        """

        filtered_cfg = {
            field.name: cfg[field.name]
            for field in fields(cls._hf_base_class)
            if field.init
        }
        return cls._hf_base_class(**dict(filtered_cfg))     # pylint: disable=not-callable


@dataclass
class HydraTrainingArguments(TrainingArguments, HydraToHFConverterMixIn):
    """The enums in the Huggingface TrainingArguments have their values, not their members as the default setting. This
        is actually an error, for example values can be duplicated while members cannot.

    If feeding TrainingArguments directly to Hydra's ConfigStore, OmegaConf throws a validation error that the enum
    default values are set to something that isn't a member of that enum. This may seem annoying but this is, in fact,
    the correct behavior.

    This really should be fixed in the transformers library.
    """

    _hf_base_class = TrainingArguments

    evaluation_strategy: IntervalStrategy = IntervalStrategy.NO
    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR
    logging_strategy: IntervalStrategy = IntervalStrategy.STEPS
    save_strategy: IntervalStrategy = IntervalStrategy.STEPS
    optim: OptimizerNames = OptimizerNames.ADAMW_HF
    hub_strategy: HubStrategy = HubStrategy.EVERY_SAVE


@dataclass
class HydraSeq2SeqTrainingArguments(Seq2SeqTrainingArguments, HydraToHFConverterMixIn):
    """The enums in the Huggingface Seq2SeqTrainingArguments have their values, not their members as the default
    setting. This is actually an error, for example values can be duplicated while members cannot.

    If feeding Seq2SeqTrainingArguments directly to Hydra's ConfigStore, OmegaConf throws a validation error that the
    enum default values are set to something that isn't a member of that enum. This may seem annoying but this is, in
    fact, the correct behavior.

    This really should be fixed in the transformers library.
    """

    _hf_base_class = Seq2SeqTrainingArguments

    evaluation_strategy: IntervalStrategy = IntervalStrategy.NO
    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR
    logging_strategy: IntervalStrategy = IntervalStrategy.STEPS
    save_strategy: IntervalStrategy = IntervalStrategy.STEPS
    optim: OptimizerNames = OptimizerNames.ADAMW_HF
    hub_strategy: HubStrategy = HubStrategy.EVERY_SAVE
