#!/usr/bin/env python
"""CLI for retino_pypeline preprocessing."""

import hydra
from omegaconf import DictConfig, OmegaConf

from patch_denoise.bindings.nipype import DenoiseParameters

from retino_pypeline.workflows.preprocessing.scenarios import (
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
        sub_id=cfg.sub,
        denoise_str=DenoiseParameters.get_str(**OmegaConf.to_container(cfg.denoiser)),
        nipype_config=OmegaConf.to_container(cfg.nipype),
        plugin=cfg.nipype_plugin.name,
        plugin_args=OmegaConf.to_container(cfg.nipype_plugin.args),
    )


if __name__ == "__main__":
    main()
