# GP Evaluation Profiling
This repository provides a means to profile (i.e., benchmark) the evaluation methodologies given by some genetic programming 
(GP) tools. The evolutionary mechanisms provided by the GP tools 
are *not* included when profiling——only mechanisms for calculating
"fitness."

This repository was created for the EuroGP 2023 conference paper 
"Using FPGA Devices to Accelerate Tree-Based Genetic Programming: A Preliminary Exploration with Recent Technologies," by Crary et al., 
which compared the evaluation performance of an initial FPGA-based GP hardware accelerator with that of the GP software tools *DEAP* (version 1.3), *TensorGP* (Git revision 09e6d04), and *Operon* (Git revision 9e7ee4e).

## Included Tools

A means for profiling for the following GP tools is given:

- [DEAP](https://github.com/DEAP/deap) - for the original paper, 
click [here](http://vision.gel.ulaval.ca/~cgagne/pubs/deap-gecco-2012.pdf).
- [TensorGP](https://github.com/AwardOfSky/TensorGP) - for the original paper,
click [here](https://cdv.dei.uc.pt/wp-content/uploads/2021/04/baeta2021tensorgp.pdf).
- [Operon](https://github.com/heal-research/operon) - for the original paper,
click [here](https://dl.acm.org/doi/pdf/10.1145/3377929.3398099).

Source code for the FPGA accelerator is not provided at this time, although the architecture is described at length in the aforementioned paper, "Using FPGA Devices to Accelerate Tree-Based Genetic Programming: A Preliminary Exploration with Recent Technologies."


## Profiling
By default, the repository already contains the results
published in the relevant conference paper.
These results are contained in the `experiment/results` directory and
can be viewed with the `experiment/tools/stats.ipynb` Jupyter Notebook file.

If so desired, after successfully completing installation (as described below), you may run 
the entire profiling suite by executing the following within a shell 
program, after having navigated to the repository directory within the shell:

```
cd experiment
bash run.sh
```

After the `run.sh` script fully executes, to view some relevant statistics, run the Jupyter Notebook file given by the path `experiment/tools/stats.ipynb`.

## Installation instructions

The following has been verified via CentOS 7. It is likely that other Linux distributions are supported, but it is unlikely that Windows and MacOS operating systems are readily supported.

### Prerequisites
- Ensure that some Conda package management system 
(e.g., [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) 
is installed on the relevant machine.
- Download the latest software release from GitHub, available [here](https://github.com/christophercrary/conference-eurogp-2023/releases). Ignore the `data.tar.gz` file for now.

Upon extracting the source code, set up the relevant Conda environment
and tools by executing the following within
a shell program, after having navigated to the repository directory
within the shell:

```
conda env create -f environment.yml
conda activate conference-eurogp-2023
pip install -r requirements.txt
bash install.sh
```

To finish installation, extract and copy the contents of the `data.tar.gz` file from the software release (i.e., the one folder and three `.pkl` files) and paste them within the `experiment/results` folder. These contents provide the random programs, inputs, and outputs utilized by the experiments.

**NOTE:** After copying the contents of the `data.tar.gz` file to the `experiment/results` folder, you may need to change file permissions for the relevant `.pkl` files. One way of doing so is by executing the following:

```
chmod 755 experiment/results/*.pkl
```

**NOTE:** If using a CPU from the Intel Skylake series (like in the conference paper), then you may need to specify this particular CPU architecture in the compilation settings for Operon before running the `bash install.sh` command listed above. To do so, comment out line 501 in `experiment/tools/operon/custom/CMakeLists.txt` and uncomment line 502.

**NOTE:** If using an Nvidia GPU (like in the conference paper), you may need to ensure that `tensorflow` can successfully utilize a GPU within the `conda` environment by prepending the following CUDA paths (or something similar) to the `$LD_LIBRARY_PATH` environment variable:

```
export LD_LIBRARY_PATH=$CONDA_PREFIX_1/pkgs/cudatoolkit-11.2.2-hbe64b41_10/lib:$CONDA_PREFIX_1/envs/tensorgp-test/lib:$LD_LIBRARY_PATH
```

If the above export command is executed, you will likely need to restart your shell to reset the `$LD_LIBRARY_PATH` environment variable after running any experiments.
