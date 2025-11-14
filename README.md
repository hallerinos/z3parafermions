# Code Supplement: "Phases of Quasi-One-Dimensional Fractional Quantum (Anomalous) Hall -- Superconductor  Heterostructures"

This repository contains the DMRG code used for the corresponding manuscript, available on [arXiv](https://arxiv.org/abs/2510.26686).

It relies on the ITensor library and contains custom dmrg routines (`dmrg0.jl` and `dmrg1.jl`) which are used in `optimize.jl` to optimize the matrix product state Ansatz.

Parameters for the simulation are contained in `cfg/default.json` and the code can be run by invoking `julia main.jl <path/to/cfg.json>`
