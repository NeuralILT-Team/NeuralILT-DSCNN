#!/bin/bash
# HPC convenience aliases for NeuralILT-DSCNN
#
# Source this file on the HPC login node:
#   source scripts/hpc_aliases.sh
#
# Or add to your ~/.bashrc:
#   echo 'source ~/NeuralILT-DSCNN/scripts/hpc_aliases.sh' >> ~/.bashrc

# ─── Job management ──────────────────────────────────────────────────
alias jobs='squeue -u $USER'
alias myjobs='squeue -u $USER --format="%.8i %.20j %.8T %.10M %.6D %R"'
alias killall='scancel -u $USER'

# ─── Quick submit ────────────────────────────────────────────────────
alias ilt-setup='bash scripts/run_hpc.sh setup'
alias ilt-download='bash scripts/download_data.sh MetalSet'
alias ilt-verify='python scripts/verify_env.py'
alias ilt-validate='python scripts/validate_pipeline.py'
alias ilt-run='sbatch scripts/run_hpc.sh'
alias ilt-baseline='sbatch scripts/run_hpc.sh baseline'
alias ilt-dscnn='sbatch scripts/run_hpc.sh dscnn'
alias ilt-eval='sbatch scripts/run_hpc.sh eval'
alias ilt-generalize='sbatch scripts/run_hpc.sh generalize'
alias ilt-analyze='python scripts/analyze_data.py all'

# ─── Log viewing ─────────────────────────────────────────────────────
alias lastlog='ls -t logs/slurm_*.out 2>/dev/null | head -1 | xargs tail -f'
alias lasterr='ls -t logs/slurm_*.err 2>/dev/null | head -1 | xargs tail -f'
alias alllogs='ls -lt logs/slurm_*.out 2>/dev/null | head -10'
alias clearlogs='rm -f logs/slurm_*.out logs/slurm_*.err && echo "Logs cleared"'

# ─── Cleanup ─────────────────────────────────────────────────────────
alias cleandata='rm -rf data/processed/MetalSet data/processed/StdMetal data/processed/StdContact && echo "Processed data deleted. Will re-preprocess on next run."'
alias cleanvenv='rm -rf venv && echo "venv deleted. Run ilt-setup to recreate."'
alias cleanall='rm -rf venv .wheels logs/slurm_* data/processed && echo "Cleaned venv, wheels, logs, and processed data"'

# ─── Results ─────────────────────────────────────────────────────────
alias results='ls -la results/ 2>/dev/null'
alias checkpoints='ls -la results/checkpoints/*/best_model.pt 2>/dev/null'

# ─── GPU node interactive session ────────────────────────────────────
alias gpunode='srun -p gpu --gres=gpu -n 1 -N 1 -c 4 --mem=16G --time=01:00:00 --pty /bin/bash'

# ─── Cluster info ────────────────────────────────────────────────────
alias nodes='sinfo -N -l'
alias gpus='sinfo -p gpu -N -l'
alias quota='df -h /home/$USER'

echo "NeuralILT HPC aliases loaded. Commands:"
echo "  jobs / myjobs     — check job status"
echo "  killall           — cancel all your jobs"
echo "  ilt-setup         — one-time environment setup"
echo "  ilt-verify        — verify all deps are working"
echo "  ilt-validate      — test full pipeline"
echo "  ilt-run           — submit full pipeline"
echo "  ilt-baseline      — train baseline only"
echo "  ilt-dscnn         — train DS-CNN only"
echo "  lastlog / lasterr — tail latest log/error"
echo "  clearlogs         — delete all log files"
echo "  gpunode           — get interactive GPU session"
