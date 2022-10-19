from transformers.utils import logging
from transformers.integrations import TensorBoardCallback, rewrite_logs


logger = logging.get_logger(__name__)


class CAITensorboardCallback(TensorBoardCallback):
    """Our own Tensorboard callback that supports additional features:

    - Currently just outputting text as well as scalars on log
    """

    @staticmethod
    def replace_in_trainer(trainer):
        trainer.callback_handler.remove_callback(TensorBoardCallback)
        trainer.callback_handler.add_callback(CAITensorboardCallback)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            logs = rewrite_logs(logs)
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
                elif isinstance(v, str):
                    self.tb_writer.add_text(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()
