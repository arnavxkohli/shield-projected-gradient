# FYP Code

This repository serves as the codebase for my Final Year Project titled "Improving Backpropagation in the PiShield package". It contains the figures, logs and metrics collected over the course of the project, as well as the code used to generate these results.

## Virtual Environments

It is recommended to use a virtual environment for this project to avoid conflicts with other packages. You can create a virtual environment using the following command:

```bash
python -m venv venv
```

Then activate the virtual environment:

### On macOS/Linux

```bash
source venv/bin/activate
```

### On Windows

```bash
venv\Scripts\activate
```

Note that all testing was done on a Linux machine.

## PiShield Repository

Due to changes made to the PiShield package, this repository is dependent on a fork of the original PiShield repository. The fork can be found here: [PiShield Fork](https://github.com/arnavxkohli/PiShield).

**It is vital to use this fork for the code to work correctly.**

This can be cloned and installed using the following command:

```bash
git clone https://github.com/arnavxkohli/PiShield.git PiShield
pip install PiShield/
```

In addition to this, the remaining relevant packages are included in the `requirements.txt` file. To install these, run:

```bash
pip install -r requirements.txt
```

## Data Download

Datasets for this project can be downloaded from the following link:
[Data Download Link](https://drive.google.com/file/d/1gZ0HUO8owebGXnaeFw_1mRcmSpYQuM6v/view?usp=sharing)

For convenience, a script is provided to download the datasets directly. This can be run from the **root directory** of the repository with:

```bash
chmod +x scripts/setup_data.sh
scripts/setup_data.sh
```

## Reproducing Results

As presented in the report, the key metrics measured included the root mean square error (RMSE), constraint satsifaction rate and the gradient alignment. CSV files for all the results can be found in the `out` directory. To reproduce the results, the following can be run from the **root directory**:

```bash
chmod +x scripts/run_tests.sh
scripts/run_tests.sh

chmod +x scripts/run_grad_analysis.sh
scripts/run_grad_analysis.sh
```

The plots can be generated with the notebook `src/plot_rmse.ipynb`. These are included in `figures/` for convenience.

## Presentation Link

For those carrying out evaluation, the presentation can be found [here](https://imperiallondon-my.sharepoint.com/:p:/g/personal/sk1421_ic_ac_uk/EfYXjAhxklpFkQN_O5ERJWMB3smAnRib4Dvs8wzsgZYL-w?e=u9fDm8)

**Note**: This section was added after the report deadline, and is not relevant to the code related to the report, it is presented in case it is helpful in providing context to the project, alongside the report.
