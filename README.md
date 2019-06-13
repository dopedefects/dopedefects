# dopedefects
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://travis-ci.com/dopedefects/dopedefects.svg?branch=master)](https://travis-ci.com/dopedefects/dopedefects)
[![Coverage Status](https://coveralls.io/repos/github/dopedefects/dopedefects/badge.svg?branch=master)](https://coveralls.io/github/dopedefects/dopedefects?branch=master)

#DopeDefects

Authors: **Ryan Beck**, **Lauren Koulias**, **Linnette Teo**

----

### Overview

DopeDefects is an open source python package that aims to predict the enthalpy of formation as well as the charge transition levels of several varous defects embedded in various Cd / Chalcogenide crystals.  


----

### Current Features
DopeDefects is able to read in a series of vasp POSCAR files and use those files along with the directory structure in order to determine the defect type, the bond lengths and angles to the atoms surrounding the defected atom, as well as calculate the coulomb matrix for the cell.  The data building will be able to add the cell information into a pandas dataframe generated from csv files containing atomic information about the cells located within the POSCAR files.  After the data has been read in, it will be saved out to file, and can be read in to resume between sessions.

----

### Configuration

#### Requirements:
 * Python version 3.6.7 or later
 * git

#### Installation & Usage

You can execute the following ``commands`` in a bash environment to download and install this package:

1. Either use git to clone the Dopedefects repository:
  
  ``git clone https://github.com/dopedefects/dopedefects.git``

  or download and unzip the zip file:

  ``curl -O https://github.com/dopedefects/dopedefects/archive/master.zip``

2. ``cd dopedefects``

3. ``pip install -r requirements.txt``

To use the package, first you will want to extract your data.  The data extraction methods are able to save the data which has been imported into a hdf5 file so it can be read back later.  To either read in the information off the file, or to generate a new file, interact with the `init_data` function of the `data_build` file.  In python:
  ``import data_build; data_build.init_data("LOCATION/OF/DATA/FILE.hdf5", data_dir="LOCATION/OF/POSCAR/FILES", [LIST_OF_CSV_INFORMATION_FILES])
If the data file specified already exists it will be read in as default, more details and options can be found in the docs.

----

### Contributions
Any contributions to the project are welcomed.  If you discover any bugs please report them in the [issues section](https://github.com/dopedefects/dopedefects/issues).

----

###License
This projects is licensed under the [MIT license](https://github.com/dopedefects/dopedefects/blob/master/LICENSE)
