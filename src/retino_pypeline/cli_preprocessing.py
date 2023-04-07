#!/usr/bin/env python
"""CLI for retino_pypeline preprocessing."""

import hydra
import logging

from omegaconf import DictConfig, OmegaConf

from nipype import config as nipype_cfg
from patch_denoise.bindings.nipype import DenoiseParameters

from retino_pypeline.workflows.preprocessing import PreprocessingWorkflowDispatcher

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="preprocessing", version_base="1.2.0")
def main(cfg: DictConfig) -> None:
    """Run the preprocessing."""

    dispatcher = PreprocessingWorkflowDispatcher(
        base_data_dir=cfg.dataset.data_dir,
        working_dir=cfg.dataset.tmp_dir,
        sequence=cfg.dataset.sequence,
    )
    dispatcher.get_workflow(cfg.build_code)
    OmegaConf.resolve(cfg.nipype)
    dict_conf = OmegaConf.to_container(cfg.nipype)
    for key, val in dict_conf.items():
        for kkey, vval in val.items():
            nipype_cfg.set(key, kkey, str(vval))

    logger.info(
        "Resource monitoring " + ("on." if nipype_cfg.resource_monitor else "off.")
    )

    dcfg = OmegaConf.to_container(cfg)
    # Extend the configuration to allow for multiple subjects and tasks
    if dcfg["sub"] == "all":
        dcfg["sub"] = [1, 2, 3, 4, 5, 6]
    if dcfg["task"] == "both":
        dcfg["task"] = ["Clockwise", "AntiClockwise"]

    dispatcher.run(
        task=dcfg["task"],
        sub_id=dcfg["sub"],
        denoise_str=DenoiseParameters.get_str(**dcfg["denoiser"]),
        plugin=dcfg["nipype_plugin"]["name"],
        plugin_args=dcfg["nipype_plugin"]["args"],
    )


if __name__ == "__main__":
    main()
