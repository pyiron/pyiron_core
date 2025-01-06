
# Instructions to Generate the .pyiron_vasp_config

The `.pyiron_vasp_config` file is essential for configuring the paths to the VASP pseudopotential files (POTCAR files) used in pyiron_workflow's VASP nodes. The file specifies the locations of different VASP potential sets and the default potential set to be used. Below are step-by-step instructions to generate the `.pyiron_vasp_config` file.

## 1. Determine the Locations of Your VASP POTCAR Files

Before creating the `.pyiron_vasp_config` file, ensure that you know the locations of the different POTCAR sets on your system. The common VASP potential directories are `potpaw_64`, `potpaw_54`, and `potpaw_52`, but your setup may vary. The directory structure should look something like this:

```
/home/pyiron_resources_cmmc/vasp/potpaw_64/LDA
/home/pyiron_resources_cmmc/vasp/potpaw_64/GGA
/home/pyiron_resources_cmmc/vasp/potpaw_54/LDA
/home/pyiron_resources_cmmc/vasp/potpaw_54/GGA
/home/pyiron_resources_cmmc/vasp/potpaw_52/LDA
/home/pyiron_resources_cmmc/vasp/potpaw_52/GGA
```

- **LDA** and **GGA** refer to the functional types for the VASP potentials.

## 2. Create the .pyiron_vasp_config File

1. Open a terminal and navigate to your home directory:

    ```bash
    cd ~
    ```

2. Use a text editor (such as `nano` or `vim`) to create and open the `.pyiron_vasp_config` file. For example:

    ```bash
    nano .pyiron_vasp_config
    ```

3. Add the following lines to the file. Make sure to modify the paths according to your setup.

    ```ini
    # Set the default POTCAR set
    # Make sure that the default_POTCAR_set matches one of the suffixes in the vasp_POTCAR_path_*
    default_POTCAR_set = potpaw64
    
    # Path to the root directory containing the VASP pseudopotential files
    pyiron_vasp_resources = /home/pyiron_resources_cmmc/vasp
    
    # Path for different POTPAW versions (adjust these paths according to your setup)
    vasp_POTCAR_path_potpaw64 = {pyiron_vasp_resources}/potpaw_64
    vasp_POTCAR_path_potpaw54 = {pyiron_vasp_resources}/potpaw_54
    vasp_POTCAR_path_potpaw52 = {pyiron_vasp_resources}/potpaw_52
    # Note that pyiron vasp nodes can detect variants of vasp_POTCAR_path_{randomsuffix}
    # So if you want to do something with custom pseudopotentials, you can... 
    # Each of these dirs must have a "GGA" and "LDA" subdirectory structure
    # i.e. 
    # The structure should look like
    # .../vasp/potpaw_64/LDA
    # .../vasp/potpaw_64/GGA
    # .../vasp/potpaw_54/LDA etc.
    ```

4. After you have added the configuration details, save the file:
   - If you're using `nano`, press `Ctrl + O`, then `Enter` to save. Press `Ctrl + X` to exit.
   - If you're using `vim`, press `Esc`, type `:wq`, and press `Enter` to save and exit.

## 3. Verify File Permissions

Ensure that your `.pyiron_vasp_config` file and the POTCAR directories are readable by your user. Run the following command to check the file permissions of the `.pyiron_vasp_config`:

```bash
ls -l ~/.pyiron_vasp_config
```

If necessary, you can modify the permissions to ensure read access:

```bash
chmod 644 ~/.pyiron_vasp_config
```

Also, verify that you have read/copy access to the files inside the VASP resource directories (`potpaw_64`, `potpaw_54`, etc.):

```bash
ls -l /home/pyiron_resources_cmmc/vasp/potpaw_64
```

## 4. Testing Your Configuration

To verify that pyiron is correctly reading the `.pyiron_vasp_config` file, you can either check within your pyiron scripts or write a simple Python script to test the configuration:

```python
import os
from pathlib import Path

# Read the config file
config_file = Path.home().joinpath(".pyiron_vasp_config")
with open(config_file, "r") as f:
    print(f.read())
```

This should print out the contents of your `.pyiron_vasp_config`, and you can check if the paths are correctly generated.

### Additional Notes:
- **Ensure Correct Paths**: Double-check that the paths to your POTCAR directories are correct. Incorrect paths will lead to pyiron not being able to find the necessary POTCAR files for your VASP calculations.

By following these instructions, you'll have a correctly configured `.pyiron_vasp_config` file that points to the appropriate VASP pseudopotential directories.
