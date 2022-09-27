#!/usr/bin/env python3
import os
import argparse
from itertools import product

from retino.workflows.preprocessing import (
    PreprocessingWorkflowFactory,
    RealignWorkflowFactory,
)
from retino.workflows.analysis import AnalysisWorkflowFactory, FirstLevelStatFactory

DATA_DIR = "/mnt/data/dataset/data"

TMP_DIR = "/mnt/data/dataset/tmp"
SEQUENCE = "EPI3D"
DENOISERS = [
    "nordic",
    "mp-pca",
    "hybrid-pca",
    "optimal-fro",
    "optimal-nuc",
    "optimal-ope",
    "noisy",
    None,
]
PROCESS = [
    "realign",
    "realign-cached",
    "glm",
    "both",
    "stats",
    "all",
]


class NumericAction(argparse.Action):
    """Argparse action to convert string to number (int or float)."""

    def __call__(self, parser, namespace, value, option_string=None):
        try:
            val = int(value)
        except ValueError:
            try:
                val = float(value)
            except ValueError:
                raise ("string not convertible to float")
        setattr(namespace, self.dest, val)


parser = argparse.ArgumentParser(
    prog="benchmark script for patch denoising method."
    "this script execute a standard retinotopy pipeline with denoising for a set of subject, but with a single configuration of denoising"
)

parser.add_argument(
    "--dataset", default=DATA_DIR, help="location of a BIDS like dataset"
)
parser.add_argument("--tmpdir", default=TMP_DIR, help="location for temp files ")
parser.add_argument("--sequence", default=SEQUENCE, help="fMRI sequence used")
parser.add_argument("--denoiser", type=str, choices=DENOISERS)
parser.add_argument("--dry", action="store_true")
parser.add_argument("--process", choices=PROCESS, default="all")
parser.add_argument("--sub", type=int, action="extend", nargs="*")
parser.add_argument("--patch-size", type=int)
parser.add_argument("--use-phase", action="store_true")
parser.add_argument("--patch-overlap", action=NumericAction)
parser.add_argument("--n_cycles", type=int, default=9)
parser.add_argument("--TR", type=float, default=2.4)


def set_patch_values(namespace):
    ps, ovl = namespace.patch_size, namespace.patch_overlap

    if type(ovl) == float:
        ovl = int(ovl * ps)
    if ovl < 0:
        ovl = ps - ovl
    if ps <= 1:
        raise ValueError("patch shape should be > 1")
    if ovl >= ps:
        raise ValueError("Overlap greater than patch shape")

    namespace.patch_size = ps
    namespace.patch_overlap = ovl


def run_realign(namespace):
    realign_wf = RealignWorkflowFactory(namespace.dataset, namespace.tmpdir)
    realign_wf.build(namespace.sequence)
    if namespace.dry:
        print(f"dry run of realign workflow with {namespace}")
    else:
        realign_wf.run(iter_on=("sub_id", namespace.sub))


def run_preprocess(namespace):

    preprocessing_wf_factory = PreprocessingWorkflowFactory(DATA_DIR, TMP_DIR)
    workflow, _ = preprocessing_wf_factory.build(
        sequence=namespace.sequence,
        denoise=namespace.denoiser != "noisy",
        patch_shape=namespace.patch_size,
        patch_overlap=namespace.patch_overlap,
        recombination="weighted",
        mask_threshold=20,
        realign_cached=namespace.process in ["realign-cached", "both"],
        use_phase=namespace.use_phase,
    )
    if namespace.dry:
        print(f"dry run of preprocessing workflow with namespace {namespace}")
    elif namespace.denoiser == "noisy":
        preprocessing_wf_factory.run(
            workflow, iter_on=("sub_id", namespace.sub), plugin="MultiProc"
        )
    else:
        preprocessing_wf_factory.run_with_denoiser(
            workflow, sub_ids=namespace.sub, denoise_methods=[namespace.denoiser]
        )


def run_glm(namespace):
    analysis_factory = AnalysisWorkflowFactory(
        namespace.dataset,
        namespace.tmpdir,
        n_cycles=namespace.n_cycles,
        TR=namespace.TR,
    )
    if namespace.denoiser == "noisy":
        wf = analysis_factory.build(
            namespace.denoiser,
        )

    else:
        wf = analysis_factory.build(
            f"{namespace.patch_size}_{namespace.patch_overlap}w"
        )
    analysis_factory.run(
            wf,
            iter_on=[
                ("sub_id", namespace.sub),
                ("denoise_method", [namespace.denoiser]),
            ],
            sequence=namespace.sequence,
            plugin="MultiProc",
        )
def run_stats(namespace):

    stats_factory = FirstLevelStatFactory(
        namespace.dataset,
        namespace.tmpdir,
    )
    if namespace.denoiser == "noisy":
        wf = stats_factory.build(
            namespace.denoiser,
        )

    else:
        wf = stats_factory.build(
            f"{namespace.patch_size}_{namespace.patch_overlap}w"
        )
    stats_factory.run(
            wf,
            iter_on=[
                ("sub_id", namespace.sub),
                ("denoise_method", [namespace.denoiser]),
            ],
            sequence=namespace.sequence,
            plugin="MultiProc",
        )



if __name__ == "__main__":

    namespace = parser.parse_args()

    if namespace.denoiser == "noisy" or namespace.denoiser is None:
        namespace.denoiser = "noisy"
        namespace.use_phase = False
    else:
        set_patch_values(namespace)
    print(namespace)

    if namespace.process == "realign":
        run_realign(namespace)
    elif namespace.process == "realign-cached":
        run_preprocess(namespace)
    elif namespace.process == "glm":
        run_glm(namespace)
    elif namespace.process == "both":
        run_preprocess(namespace)
        run_glm(namespace)
    elif namespace.process == "stats":
        run_stats(namespace)
    elif namespace.process == "all":
        run_preprocess(namespace)
        run_glm(namespace)

    else:
        raise ValueError(f"Unknown process configuration '{namespace.process}'")