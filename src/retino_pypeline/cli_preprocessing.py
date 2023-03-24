"""CLI for retino_pypeline preprocessing."""

import hydra
from omegaconf import DictConfig, OmegaConf

from retino_pypeline.interfaces.denoise import DenoiseParameters
from retino_pypeline.workflows.preprocessing import (
    NoisePreprocManager,
    RetinotopyPreprocessingManager,
)


@hydra.main(config_path="conf", config_name="preprocessing")
def main(cfg: DictConfig) -> None:
    """Run the preprocessing."""

    if cfg.run_noise:
        noise_prep_mgr = NoisePreprocManager(cfg.dataset.data_dir, cfg.dataset.tmp_dir)
        wf = noise_prep_mgr.get_workflow()
        noise_prep_mgr.run(
            wf,
            multi_proc=True,
            sub_id=cfg.dataset.sub,
            task=cfg.dataset.task,
            sequence=cfg.dataset.sequence,
        )

    prep_mgr = RetinotopyPreprocessingManager(cfg.dataset.data_dir, cfg.dataset.tmp_dir)

    wf = prep_mgr.get_workflow(cfg.build_code)
    prep_mgr.run(
        wf,
        multi_proc=True,
        sub_id=cfg.dataset.sub,
        task=cfg.dataset.task,
        sequence=cfg.dataset.sequence,
        denoise_str=DenoiseParameters.get_str(**OmegaConf.to_container(cfg.denoiser)),
    )


if __name__ == "__main__":
    main()
