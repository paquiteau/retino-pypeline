"""CLI for retino_pypeline preprocessing."""

import hydra
from omegaconf import DictConfig, OmegaConf

from patch_denoise.bindings.nipype import DenoiseParameters

from retino_pypeline.worklows.preprocessing.scenario import (
    PreprocessingWorkflowDispatcher,
)


@hydra.main(config_path="conf", config_name="preprocessing")
def main(cfg: DictConfig) -> None:
    """Run the preprocessing."""

    dispatcher = PreprocessingWorkflowDispatcher(
        base_data_dir=cfg.dataset.data_dir,
        working_dir=cfg.dataset.tmp_dir,
        sequence=cfg.dataset.sequence,
    )
    dispatcher.get_workflow(cfg.build_code)

    dispatcher.run(
        task=cfg.task,
        sub_id=cfg.sub_id,
        denoise_str=DenoiseParameters.get_str(OmegaConf.to_container(cfg.denoise)),
        plugin=cfg.plugin.name,
        plugin_args=OmegaConf.to_container(cfg.plugin.args),
    )


if __name__ == "__main__":
    main()
