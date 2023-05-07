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

1. Run `python src/GPFramework/launchGTMOEP.py templates/<input-file>` from the top-level directory.  Emade is now running, and writing its output to your local disk. Check your current working directory for "masterXXXXX.out/.err" and "workerXXXXX.out/.err" files. Note that it may take several minutes for EMADE to write its first output to masterXXXXX.out and several more for workerXXXXX.out.
2. Note that launchGTMOEP.py is a script to launch didLaunch.py. If you need to manually launch didLaunch.py for debugging purposes, see the "Running a SQLlite Run of Emade" section below.

## Running a SQLlite Run of EMADE:
**Note: the recommended way to run emade on a non-embedded device is using MySQL, as detailed above**

If you **don't have** a database already, run `python src/GPFramework/didLaunch.py myPickleFile#####.dat -n <#-workers>` where `-n <#-workers>` is **not** required for the master worker. This will generate a local SQLite database by default with a name like `EMADE_07-21-2017_10-18-41.db` which follows the format `EMADE_<Date>_<Time>.db`.
        * If you already **have** a database you want to use (every normal worker needs a database), run `python src/GPFramework/didLaunch.py myPickleFile#####.dat -n <#-workers> -d dialect[+driver]://user:password@host/dbname -r/-nr [-ma]` where `[]` parts are optional.
        * `-n <#-workers>` is **not** required for the master worker. `-r/-nr` is **not** required for normal workers. Both will be ignored if added in each case.
        * Read [SQLAlchemy Database Engine Format Docs](http://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine) to figure out the right database connection string.
        * Read sublist above to learn how to save terminal output.
    * Example Master Worker: `python src/GPFramework/didLaunch.py myPickleFile13476.dat -d mysql://scott:tiger@localhost/testdb -r -ma` to connect to a MySQL database or `python src/GPFramework/didLaunch.py myPickleFile8086.dat -ma` for a local SQLite database.
    * Example Normal Worker: `python src/GPFramework/didLaunch.py myPickleFile9452.dat -n 4 -d sqlite:///EMADE_07-21-2017_10-18-41.db`
    * From reference from `didLaunch.py`:
        * `-d, --database` specifies your database connection string
        * `-ma, --master` is included if this worker is the master worker.
        * If you're running the master worker, you must append `[-r or -nr]` or equivalently `[--reuse or --no-reuse]` with `-d` to determine whether to reuse the tables generated by EMADE or to start from scratch.
        * `-n <#-workers>` is necessary for normal workers (will default to 1 if not included) but not needed for the master worker.

## Tutorials
* **Genetic Algorithms/Genetic Programming**
    * EMADE uses DEAP, or Distributed Evolutionary Algorithms in Python, to perform genetic programming.
    * First read the *First Steps* and *Basic Tutorials* in the [DEAP Documentation](http://deap.readthedocs.io/en/master/).
    * Genetic Algorithms - Understand and run [One Max Problem](http://deap.readthedocs.io/en/master/examples/ga_onemax.html).
    * [Genetic Programming](http://deap.readthedocs.io/en/master/tutorials/advanced/gp.html) - Understand and run [Symbolic Regression](http://deap.readthedocs.io/en/master/examples/gp_symbreg.html).
    * Check out other [examples](http://deap.readthedocs.io/en/master/examples/index.html).
    * We have provided lectures and labs to help you get started.
        * [**Lecture 1: Genetic Algorithm**](https://docs.google.com/presentation/d/1245i2_d1AD8PKkwE3DMTBeH9cuQX4B2LZul3ljzLvP8/edit#slide=id.p)
        * [**Lecture 2: Genetic Programming**](https://docs.google.com/presentation/d/1d04gRdeCVXXeRglz39dqEk95SkGi6TYT9c-WvElIeM8/edit?usp=sharing)
        * [**Lab 1: Genetic Algorithms with DEAP**](https://github.gatech.edu/emade/emade/blob/master/notebooks/Lab%201%20-%20Genetic%20Algorithms%20with%20DEAP.ipynb)
        * [**Lab 2: Genetic Programming and Multi-Objective Optimization**](https://github.gatech.edu/emade/emade/blob/Database/notebooks/Lab%202%20-%20Genetic%20Programming%20and%20Multi-Objective%20Optimization.ipynb)
        * [**Preprocessing Data**](https://github.gatech.edu/emade/emade/blob/Database/notebooks/Preprocessing%20Data.ipynb)
* **scikit-learn**
    * scikit-learn is the machine learning library for Python that underlies many of the primitive learner functions.
    * Follow the examples and documentation [here](http://scikit-learn.org/stable/) and try out common algorithms like Linear/Logistic Regression, KNN, SVM, and k-Means among others.
* **Other Useful Libraries**
    * Understand the fundamentals of [numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html).
    * Learn about parallel computing and multiprocessing with [SCOOP](http://scoop.readthedocs.io/en/0.7/).
