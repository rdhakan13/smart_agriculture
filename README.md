# :rocket: Multiple nutrient sensing for smart agriculture

Contributors: John White, Thomas Wilson, Raj Dhakan, Ross Duncan

## :page_facing_up: Introduction

Near-Infrared spectroscopy (NIRS) is a technology used in 'precision agriculture'[^1], classified as a remote sensing technique[^2], which can produce data to help maintain optimal nutrient levels in potato plants[^3]. Using the reflectivity readings of leaf samples produced by NIRS (as characteristics) together with measurements of nutrients (as targets), supervised machine learning (ML) techniques can be used to predict plant nutrient levels faster, more efficiently and cost-effectively when compared to traditional laboratory tests. NIRS produces high-dimensional reflectivity readings characterised by $p \gg n$ (with $p$ features and $n$ samples), and there are multiple nutrients of interest in the health/yield of potato plants. The purpose of this study is to extend the research of Abukmeil et al.[^3], which focused on the effectiveness of using floral spectroscopy to predict plant nutrient concentrations using a Lasso multi-linear regression (MLR) approach, to:

- Analyse the effect of data preprocessing methods on spectral data and its effect on predictive performance.
- Establish the impact of the elected data processing strategy and its suitability against multivariate models commonly used in similar settings.

Initially, the data is parsed through a preprocessing pipeline featuring baseline correction of a dataset and extracting the features of interest, which is compiled over multiple seasons to maximise generalisation. A univariate MLR baseline is used to assess the performance of basic data reduction techniques such as binning and Principle Component Analysis (PCA), along with more tailored techniques of baseline correction coupled with peak feature extraction. The resulting pipeline was compared to the performance of a multivariate application of a Partial Least Squares Regression (PLS or PLSR) model, Multi Task Lasso (MTL), Multi Task Elastic Net (MTEN) and Random Forest (RF) models. The effectiveness of the pipeline for fresh and dried sample modes is examined separately, as they present different reflective signals.

The findings of the research in this code repository is documented in full on this [paper](reports/docs/DSMP_Written_Report_Group_O2.pdf).

## :floppy_disk: Dataset

The data in this study consist of NIRS reflectance readings (features) and 14 (macro and micro) nutrient concentrations (targets), where nutrients are measured in parts per million (PPM) or percentage (PCT) as described by the matrix in the table below.

Reflectance is measured over a spectrum of wavelengths between 400nm–2500nm at either 0.5nm or 1nm intervals. The sample size `n = 674` (excluding null records) can have features `p = 4200` (at the higher resolution of 0.5nm), defining the dataset as high-dimensionality low-sampling size (HDLSS). The data is further categorised by its season (of which there are 4) and the two sampling modes (dried and fresh).

|         | Macro                  | Micro                 |
|---------|------------------------|-----------------------|
| **PPM** | -                      | Al, B, Cu, Fe, Mn, Zn |
| **PCT** | Ca, Mg, N, P, K, S     | Cl, Na                |

*Dataset owners Abukmeil et al.[^3].

## :open_file_folder: Project Structure
```
dsmp-2024-groupo2/
│
├── configs/                      # Contains yaml config files used for full_model_template.py
|
├── data/                         # Folder containing datasets
│   ├── als_corrected_dried/      # Placeholder for ALS baseline corrected dried data
│   ├── als_corrected_dried/      # Placeholder for ALS baseline corrected fresh data
│   ├── leaf_samples/             # Placeholder for leaf sample data
|   └── element_groups.csv/       # Metadata for the units of measurement for each target
│
├── images/                       # Folder containing images
│   ├── baseline/                 # Contains images for baseline corrected data
│   ├── corrected_spectra/        # Contains images for corrected spectra
│   ├── EDA/                      # Contains images for exploratory data analysis
|   └── missing_data/             # Contains images for summary of missing data
|
├── reports/                      # Folder containing data for final report
│   ├── docs/                     # Contains final report documenting results of the study
|   └── results/                  # Contains excel files for all experiments done for each model
|
├── spectroscopy/                 # Main project source code
|   ├── notebooks/                # Jupyter notebooks for exploratory data analysis (EDA) and prototyping
│   ├── scripts/                  # Scripts for running experiments and models
│   └── src/                      # Source code for preprocessing and modeling
│       ├── common/               # Shared constants and utility functions
│       ├── models/               # Machine learning models
│       ├── postprocessing/       # Postprocessing tasks such as plots and analysing metrics
│       └── preprocessing/        # Preprocessing tasks such as baseline correction, target scaling, imputation etc
│
├── results/                      # Folder for storing results and outputs
│   ├── plots/                    # Generated plots (e.g., scree plots, cumulative variance plots)
│   └── metrics/                  # Evaluation metrics for models
|
├── .gitignore                    # Git ignore file
├── environment.yml               # Conda environment configuration file
├── Makefile                      # Automation tasks for environment setup and script 
├── README.md                     # Project documentation and usage instructions
└── setup.py                      # Python package setup file
```

## :gear: How to Run

This project uses a `Makefile` to automate tasks related to managing a Conda environment, code clean-up and running scripts.

### Prerequisites

Before using the `Makefile`, make sure you have the following installed on your system:

1. **Conda**: The `Makefile` relies on Conda to manage virtual environments. You can download and install Conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
   
2. **Make**: You need the `make` utility to execute the commands in the `Makefile`. Most Linux/macOS systems come with `make` preinstalled. On Windows, you can install `make` by downloading the wizard from [GCC for Windows](https://sourceforge.net/projects/gnuwin32/files/make/3.81/make-3.81.exe/download?use_mirror=altushost-swe&download=). Install `make` by running the wizard and copying the path of executable into PATH variable under 'Edit environmental variables for your account' in control panel. You can find the path of the executable by running `where make` in the terminal.

3. **Python**: Ensure that you have Python installed (though Conda will handle this when creating the environment).

### Usage Instructions

The `Makefile` includes several targets for managing the Conda environment. Use `make` followed by the target name to perform the desired action.

### Available Targets:

**Create a new Conda environment**:
   This target creates a new Conda environment from the `environment.yml` file and installs the specified version of Python:
   ```
   make create-env
   ```

**Update an existing Conda environment**:
   This target updates the Conda environment based on changes in the `environment.yml` file. Update the dependencies as needed with python packages and respective versions:
   ```
   make update-env
   ```

**Remove the Conda environment**:
    This target removes the Conda environment entirely:
```
make remove-env
```


To activate the environment:
```
conda activate DSMP
```

To deactivate the environment:
```
conda deactivate
```

To run scripts:
1. Set config file (use [config_template](configs/config_template.yml) for guidance) as environment variable:
```
set SPECTROSCOPY_CONFIG=lasso.yml
```
2. Run model
```
make run-model
```

## :books: References
[^1]: A. M. Cavaco, A. B. Utkin, J. Marques da Silva, and R. Guerra, “Making sense of light: The use of optical spectroscopy techniques in plant sciences and agriculture,” Applied Sciences, vol. 12, no. 3, 2022.

[^2]: H. Jafarbiglu and A. Pourreza, “A comprehensive review of remote sensing platforms, sensors, and applications in nut crops,” Computers and Electronics in Agriculture, vol. 197, p. 106844, 2022.

[^3]: R. Abukmeil, A. A. Al-Mallahi, and F. Campelo, “New approach to estimate macro and micronutrients in potato plants based on foliar spectral reflectance,” Computers and Electronics in Agriculture, vol. 198, pp. 1070–74, 2022.