#!/usr/bin/env python3
"""Base configuration for the cli tools."""

import argparse
import os

BASE_DIR = "/neurospin/optimed/pierre-antoine/dataset/"
DATA_DIR = BASE_DIR + "data"
TMP_DIR = BASE_DIR + "tmp"
TASKS = ["Clockwise", "AntiClockwise"]
SEQUENCES = ["EPI3D"]


def base_parser(prog):
    """Base Parser configuration."""
    parser = argparse.ArgumentParser(prog=prog)

    parser.add_argument(
        "--basedir",
        default=BASE_DIR,
        help="Base location for dataset and tmp folder.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="location of a BIDS like dataset",
    )
    parser.add_argument(
        "--tmpdir",
        default=None,
        help="location for temp files",
    )
    parser.add_argument(
        "--sequence",
        help="fMRI sequence used",
        nargs="+",
        default=SEQUENCES,
    )
    parser.add_argument(
        "--denoise-str",
        type=str,
        nargs="+",
        help="denoiser config string",
    )
    parser.add_argument(
        "--build-code",
        type=str,
        nargs="+",
        help="build code for the preprocessing pipeline, eg `Rd`, `rD`, `rd` etc...",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="dry mode:  the workflow is not run, instead exec graph is produced",
    )
    parser.add_argument(
        "--sub",
        type=int,
        nargs="+",
        default=list(range(1, 7)),
        help="list of subject to process.",
    )
    return parser


def get_namespace(parser):
    """Get argument and sanitize a bit."""
    ns = parser.parse_args()
    if ns.basedir is None:
        ns.dataset = ns.dataset or DATA_DIR
        ns.tmpdir = ns.tmpdir or TMP_DIR
    else:
        ns.dataset = ns.dataset or os.path.join(ns.basedir, "data")
        ns.tmpdir = ns.tmpdir or os.path.join(ns.basedir, "tmp")
    return ns
