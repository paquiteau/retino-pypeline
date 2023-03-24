"""CLI for retino_pypeline analysis."""


import hydra
from omegaconf import DictConfig

from retino_pypeline.interfaces.denoise import DenoiserParameters

from retino_pypeline.workflows.analysis import (
    FirstLevelStats,
    RetinoAnalysisWorkflowManager,
)


@hydra.main(config_path="conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    """Run the analysis."""

    if cfg.tsnr:
        mgr = FirstLevelStats(cfg.dataset.data_dir, cfg.dataset.tmp_dir)
        wf = mgr.get_workflow()
        mgr.run(
            wf,
            multi_proc=True,
            sub_id=cfg.sub,
            preproc_code=cfg.build_code,
            denoise_str=DenoiserParameters.get_str(cfg.denoiser),
        )
    else:
        mgr = RetinoAnalysisWorkflowManager(cfg.dataset.data_dir, cfg.dataset.tmp_dir)
        wf = mgr.get_workflow(n_cycles=9, threshold=0.001)
        wf.inputs.input.volumetric_tr = 2.4
        mgr.run(
            wf,
            multi_proc=True,
            sub_id=cfg.sub,
            sequence=cfg.dataset.sequence,
            denoise_str=DenoiserParameters.get_str(cfg.denoiser),
        )


if __name__ == "__main__":
    main()
