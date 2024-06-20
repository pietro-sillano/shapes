import subprocess
import os
import argparse
import numpy as np
# import matplotlib.pyplot as plt


def create_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def main():
    ##### ARGS #######
    # PATH
    # LAMMPS INPUT FILE
    #
    #
    #
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        help="absolute simulation folder path",
        default='/home/psillano/scratch/membrane-lammps/Oneparticlethick/solvent')

    opt = parser.parse_args()

    create_dir(f"{path}")

    file = open(f"{path}/{sim_name}.sh", 'w')
    output = f"""#!/bin/bash
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

"""
    file.write(output)
    file.close()

    sbatch_command = f"sbatch {path}/{sim_name}.sh"

    print(sbatch_command)

     #  Run sbatch command to submit the script
     print(DRYRUN)
      if not DRYRUN:
           try:
                subprocess.run(sbatch_command, check=True, shell=True)
                print("sbatch script submitted successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting sbatch script: {e}")


if __name__ == "__main__":
    main()
