"""CLI for retino_pypeline preprocessing."""

import hydra
from omegaconf import DictConfig

from retino_pypeline.interfaces.denoise import DenoiserParameters
from retino_pypeline.workflows.preprocessing import (
    NoisePreprocManager,
    RetinoPreprocessingManager,
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
            sub_id=cfg.sub,
            task=cfg.task,
            sequence=cfg.dataset.sequence,
        )

    prep_mgr = RetinoPreprocessingManager(cfg.dataset.data_dir, cfg.dataset.tmp_dir)

    wf = prep_mgr.get_workflow(cfg.build_code)
    prep_mgr.run(
        wf,
        multi_proc=True,
        sub_id=cfg.sub,
        task=cfg.task,
        sequence=cfg.dataset.sequence,
        denoise_str=DenoiserParameters.get_str(cfg.denoiser),
    )


if __name__ == "__main__":
    main()
