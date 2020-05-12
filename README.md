# Network Diffusion (nDiffusion) for Validating Gene Connectedness

Repository for [*nDiffusion*](https://www.worldscientific.com/doi/10.1142/9789811215636_0039).   - (will be changed later)

If you have any questions or comments, feel free to contact Minh Pham (minh.pham@bcm.edu) or Olivier Lichtarge (lichtarge@bcm.edu).

--------
## Content
 - [Download code](#download-code)
 - [Installation and download network data](#installation-and-download-network-data)
 - [Run tutorial](#run-tutorial)

--------
## Download code   - (will be changed later)
```bash
git clone https://github.com/mpham93/nDiffusion.git 
```

--------
## Installation

### Install Environment
- Requirement: python=3.5.2
```bash
conda create -n nDiffusion python=3.5.2
source activate nDiffusion
pip install -r requirements.txt
```
--------
## Run tutorial

### 1. Activate environment (skip it if you already activated)
```bash
source activate nDiffusion
```
### 2. Open src/run_Diffusion.py and edit the appropriate variables in the CHANGE HERE section. For the tutorial, keep the default
#### (1) A network file: network_fl. Format: Entity1\tEntity2\tEdge_weight. Default: toy_network.txt
#### (2) Input gene lists: geneList1_fl and geneList2_fl. Format: a column of gene names
#### (3) A result folder: result_fl. Default: '../results/'

### 3. Run run_Diffusion.py
```bash
cd src/
python run_Diffusion.py
```
### 4. Check for the outputs in the result folder
=======

