#!/bin/bash
#SBATCH --partition=compute
#SBATCH --time=00:10:00
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G #per node
#SBATCH --account=research-as-bn
#SBATCH --output={path}/out.%j
#SBATCH --error={path}/err.%j
#SBATCH --mail-user=p.sillano@tudelft.nl
#SBATCH --mail-type=ALL ##you can also set BEGIN/END

module load 2022r2
module load py-scipy
module load py-joblib

python differential_ev.py