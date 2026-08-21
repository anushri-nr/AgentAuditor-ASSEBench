#!/bin/bash
# --output must be declared before any non-#SBATCH executable line in the script. The `logs/`
# directory has to exist BEFORE you submit - SLURM opens this file the instant the job starts, not
# when the script gets to running, so `mkdir -p logs` inside the script itself is too late.
#SBATCH --job-name=agent-auditor
#SBATCH --output=logs/%x_%j.out

# Before running the script, configure LLM API keys and endpoints in .py files under AgentAuditor/tasks, along with model names
# To run an experiement, take rjudge as an example:

# Stop on the first failed stage rather than continuing on to later stages against missing/broken
# output from a crashed one - matters most on HiPerGator burst QOS, where every stage below costs
# real API calls and queue time that a silent partial-pipeline run would waste.
set -euo pipefail

# sbatch runs this script in a fresh non-interactive shell that does NOT source ~/.bashrc, so
# `python` isn't on PATH unless loaded here explicitly - see https://docs.rc.ufl.edu/quickstart/computation/
# UF RC recommends Conda over a bare venv/pip install for dependency management - see
# https://docs.rc.ufl.edu/software/conda_environments/. Path-based env shared across the iruchkin
# group - only the maintainer should `pip install`/upgrade into it (see UF RC's own caveat on
# shared envs); everyone else just activates and runs.
module load conda
conda activate /blue/iruchkin/share/conda/envs/agentauditor

# Shared across every stage below (each is its own `python` process) so they all land in the same
# per-run timings file instead of the previous single timings.json every run appended into. Under
# SLURM this is just the job ID; timer.py falls back to SLURM_JOB_ID on its own if this isn't set,
# so it's only needed here to also group stages when running this script outside of SLURM.
export AGENTAUDITOR_RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%dT%H%M%S)}"

# Sequentially run the following commands, remember to check successful completion of each step
python -m AgentAuditor rjudge preprocess
python -m AgentAuditor rjudge cluster
python -m AgentAuditor rjudge demo
python -m AgentAuditor rjudge infer_emb
python -m AgentAuditor rjudge infer
python -m AgentAuditor rjudge eval

# Notes: Only one model and one dataset can be used at a time. If you want to parallelize the process,
# just make a copy of the repo.
