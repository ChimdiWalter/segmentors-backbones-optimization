#!/bin/bash
#SBATCH --job-name=compile_paper
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/cluster/VAST/kazict-lab/e/lesion_phes/code/segmentors_backbones/Paper_3/manuscript_update_fullbudget_9leaf/slurm/compile_%j.out
#SBATCH --error=/cluster/VAST/kazict-lab/e/lesion_phes/code/segmentors_backbones/Paper_3/manuscript_update_fullbudget_9leaf/slurm/compile_%j.err

cd /cluster/VAST/kazict-lab/e/lesion_phes/code/segmentors_backbones/Paper_3/manuscript_update_fullbudget_9leaf/tex

echo "=== Compiling paper ==="
echo "pdflatex: $(which pdflatex) $(pdflatex --version | head -1)"
echo "Start: $(date)"

pdflatex -interaction=nonstopmode Latest_Paper_1_updated_fullbudget9.tex
bibtex Latest_Paper_1_updated_fullbudget9
pdflatex -interaction=nonstopmode Latest_Paper_1_updated_fullbudget9.tex
pdflatex -interaction=nonstopmode Latest_Paper_1_updated_fullbudget9.tex

if [ -f Latest_Paper_1_updated_fullbudget9.pdf ]; then
    cp Latest_Paper_1_updated_fullbudget9.pdf ../pdf/
    echo "PDF copied to pdf/"
    ls -la ../pdf/
else
    echo "ERROR: PDF not generated"
fi

echo "Finish: $(date)"
