# EMADE-NAS
This is Cameron Whaley's personal repository for EMADE/AAD.
I worked under Dr. Jason Zutty on the [Automated Algorithm Design](https://www.vip.gatech.edu/teams/vvk) VIP project at Georgia Tech. This fork uses EMADE's genetic programming engine to create Neural Network architectures for any dataset. I worked on this project from Spring 2020 -- Fall 2021, co-leading the team in my final semester. Here is that semester's [final presentation](https://docs.google.com/presentation/d/1kEOKk6Esu_CEE2FzyLRO6HDHYNDNhkNw4JH3fD7fRV0/edit?usp=sharing).

# EMADE
Evolutionary Multi-objective Algorithm Design Engine

This project is licensed under the terms of the MIT license.

## Requirements
EMADE requires Python v3.8.5

EMADE requires the following Python libraries:

* numpy
* pandas
* keras
* deap
* scipy
* psutil
* lxml
* matplotlib
* hmmlearn
* PyWavelets
* multiprocess
* sqlalchemy
* networkx
* tensorflow
* networkx
* lmfit
* cython
* scikit-image
* opencv-python (imported as cv2)
* pymysql
* mysqlclient
Optional:
* tensorflow-gpu
Required Only For Windows:
* scoop

Note: Compiling Tensorflow from source code can allow it to take advantage of more CPU instruction sets, and even integrated GPUs on some chips (e.g. Intel Xeon). 

## Installation
1. Install [Git](https://git-scm.com/). Run `sudo apt-get install git` (for Ubuntu/Debian) in the terminal. Check [here](https://git-scm.com/download/linux) for package managers on other Linux distros.
    * On Windows download the [.exe](https://git-scm.com/download/win), run it, and follow the install instructions.
    * On Windows and macOS, you may need to add Git to your PATH environment variable. Windows instructions can be found [here](https://stackoverflow.com/questions/26620312/installing-git-in-path-with-github-client-for-windows) and macOS instructions can be found [here](https://stackoverflow.com/questions/1835837/git-command-not-found-on-os-x-10-5).
2. Install [Git LFS](https://git-lfs.github.com/). You can find Linux installation instructions [here](https://help.github.com/articles/installing-git-large-file-storage/).
   * This includes running the `git lfs install` command, this must be done *prior to* the cloning of the repository from github or the data files will not be properly downloaded.
3. Run `git config --global credential.helper cache` in the terminal to reduce username and password prompts
4. Clone the [git repository](https://github.gatech.edu/emade/emade). Run `git clone https://github.gatech.edu/emade/emade` at your home directory.
    * If you struggle with authentication, try https://USERNAME@github.gatech.edu/emade/emade
5. Install [Anaconda 3](https://www.continuum.io/downloads). Read documentation for conda environment management [here](https://conda.io/docs/using/envs.html).
    * Make sure to type `yes` when it asks if you would like the 'installer to prepend the Anaconda3 install location to PATH'.
6. Close your current terminal and open a new terminal to change your default python version.
7. Run `cd emade` in the terminal.
8. Run `conda install python==3.8.5 opencv` in the terminal.
9. Run `conda install "seuptools>=57,<58"` in the terminal.
10. Install the required packages by running `conda install numpy pandas tensorflow==2.6.0 keras==2.6.0 scipy psutil lxml matplotlib PyWavelets sqlalchemy networkx cython scikit-image mysqlclient pymysql scikit-learn tensorflow-datasets==4.3 sep nltk textblob spectral` and subsequently `pip install xgboost lmfit multiprocess hmmlearn deap==1.2.2 opencv-python keras-pickle-wrapper`. Conda has superior features resolving dependencies, but not all required packages are in the standard conda repositories. Therefore, we use both. 
   * If mysqlclient fails to install due to a missing mysql_config dependency, try to install libmysqlclient-dev or search for a similar package for your OS. [e.g. on Debian use apt-cache search libmysqlclient]
   * If a recently upgraded package has created a version conflict, you can force conda to install a previous version of a package by using this syntax: conda install numpy=1.14.5
   * Install scoop as well if you're on Windows
   * If hmmlearn fails to build on Windows, install the [Microsoft 2017 build tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15), then `conda install -c conda-forge hmmlearn`
11. Run `bash reinstall.sh` (macOS/Linux) or `reinstall` (windows) in the terminal to build all the required files.
12. Install MySQL server (https://dev.mysql.com/downloads/installer/) and configure an empty SQL schema (i.e. a database with a name and login credentials, but no tables). Note that you can do a local run of EMADE with sqllite by manually running didLaunch.py as described below, but to take advantage of Emade's distributed architecture, you will need MySQL. Also, the GUI of [MySQL Workbench](https://www.mysql.com/products/workbench/) is very helpful. 
[note on Debian linux you may simply run sudo apt-get install mysql-client mysql server]

## Usage
0. Note that you will first need to populate the input_file with your database info (IP address, login, password, etc.). Examples are in the templates/ directory. Note that if you're running a local run, the string "localhost" will resolve as your local IP.

1. Run `python src/GPFramework/launch.py templates/<input-file>` from the top-level directory.  Emade is now running, and writing its output to your local disk. Check your current working directory for "masterXXXXX.out/.err" and "workerXXXXX.out/.err" files. Note that it may take several minutes for EMADE to write its first output to masterXXXXX.out and several more for workerXXXXX.out.
