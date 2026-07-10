# Installation

## with pip (recommended)

DASexplorer is published on [PyPI](https://pypi.org/project/dasexplorer/) and can be installed with `pip`, ideally into a dedicated [virtual environment](https://docs.python.org/3/library/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

=== "Latest"

    ```
    pip install dasexplorer
    ```

=== "1.x"

    ```
    pip install dasexplorer==1.x
    ```

This will automatically install compatible versions of all direct dependencies: `PyQt5`, `pyqtgraph`, `numpy`, `h5py`, and `scipy`.

!!! tip

    If you don't have prior experience with Python, we recommend reading [Installing Packages](https://packaging.python.org/en/latest/tutorials/installing-packages/) from the Python Packaging User Guide, which is a good introduction to the mechanics of Python package management and helps you troubleshoot if you run into errors.

## with git

DASexplorer is a new tool that is updated frequently. As a result, there may be minor changes every day that are not reflected in the pip version. DASexplorer can be used directly from GitHub by cloning the repository into a subfolder of your project root, which may be useful if you want to use the very latest version:

```
git clone https://github.com/sermomon/DASexplorer.git
```

Next, install DASexplorer and its dependencies with:

```
pip install -e dasexplorer
```

The `-e` flag installs the package in [editable mode](https://packaging.python.org/en/latest/tutorials/packaging-projects/), so any changes you pull from the repository are reflected immediately without reinstalling.

## Verifying the installation

To confirm that the installation was successful, you can launch the GUI by typing `dasexplorer` in your terminal:

```
dasexplorer
```

If everything is set up correctly, the DASexplorer main window should open.
