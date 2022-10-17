#!/usr/bin/env python3
"""Script to perform the analysis of data."""

from .base import base_parser


def get_parser():
    """Get parser."""
    parser = base_parser("Compute the analysis of data")
    return parser


def analyse(ns):
    """Perform analysis."""
    pass


def main_cli():
    """Run cli."""
    parser = get_parser()

    ns = parser.parse_args()
    analyse(ns)


if __name__ == "__main__":
    main_cli()
