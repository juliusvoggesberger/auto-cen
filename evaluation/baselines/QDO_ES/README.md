# Q(D)O-ES Custom Fork

This contains a changed version of Q(D)O-ES in order to be able to run the code
(in one file/command instead of separate files) with the following custom changes:

## Changes:
    - Handles different conda environments (because of the broken original setup)
    - Added the possibility to run the pipeline on a dataset_id instead of a task_id.
    - Added the possibility to run only the ensemble selection methods.
    - Added the possibility to specify the number of folds to run. (Including a single fold)
    - Added the possibility to specify the number of cores to use.
    - Added the possibility to specify the seed to use.
    - Added the possibility to delete the temporary files after running the pipeline.
    - Handles moving the output of step 3 to the benchmark folder. (For Step 4)
    - Saves the end results in a .txt file for each evaluated Ensemble Selection into "/evaluation_custom"

## Installation
For the installation, pip and conda (or mamba) are required.

1. Clone the repository
2. Create two different conda environment with the required packages:

   <details>

      <summary>First Conda environment:</summary>

          ```
            conda create -n "autosklearn_custom" python==3.8.10
            conda activate autosklearn_custom

            # Using scitkit-learn==1.0.2 works:
            pip install "ConfigSpace>=0.4.21,<0.5" "dask>=2021.12" "distributed>=2012.12" distro joblib liac-arff "numpy>=1.9.0" "pandas>=1.0" "pynisher>=0.6.3,<0.7" "pyrfr>=0.8.1,<0.9" PyYAML scikit-learn==1.0.2 "scipy>=1.7.0" setuptools "smac>=1.2,<1.3" threadpoolctl typing_extensions assembled[openml]==0.0.4 tables openml requests

            # Original doesn't work, because of mismatching scikit-learn version for auto-sklearn (>=0.24.0,<0.25.0) and assembled[openml] (==1.0.2)
            pip install "ConfigSpace>=0.4.21,<0.5" "dask>=2021.12" "distributed>=2012.12" distro joblib liac-arff "numpy>=1.9.0" "pandas>=1.0" "pynisher>=0.6.3,<0.7" "pyrfr>=0.8.1,<0.9" PyYAML "scikit-learn>=0.24.0,<0.25.0" "scipy>=1.7.0" setuptools "smac>=1.2,<1.3" threadpoolctl typing_extensions assembled[openml]==0.0.4 tables openml requests
          ```

   </details>

   <details>

      <summary>Second Conda environment:</summary>

          ```
            conda create -n "qdo" python==3.8.10
            conda activate qdo
            pip install scikit-learn==1.0.2 assembled[openml]==0.0.4 ribs==0.4.0 ConfigSpace==0.6.1
          ```

   </details>

3. Add the custom auto-sklearn fork into a folder titled `autosklearn`.

## Usage
Currently, the Parameters aren`t set in the command-line (Only set inside the main.py)

1. Change the following parameters inside the main.py file:
   - Specify the Conda environment paths (In order for the script to activate and deactivate the needed environments)
   - Specify the task_id (or dataset_id with "is_dataset=True") and its runtime in seconds
   - Set desired seeds
   - Set "should_retry = True" if the script should retry all task_id seed combinations, until they all ran successfully
   - Set "memory_limit" (per_Core) and "n_jobs" ("memory_limit" is somehow bugged and can throw random memory errors...)
2. Run main.py
3. Results are saved inside "/evaluation_custom/" in .txt files (One file per task_id with each ensemble technique)
