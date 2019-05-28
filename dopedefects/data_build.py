"""
Construct the data matrix from the structure extract and structure
properties and fold them into the existing .csv files and return a data
object that can be resumed.
"""
import h5py
import numpy as np
import os
import pandas
try:
    import structure_extract
    import structure_properties
except:
    import dopedefects.structure_extract as structure_extract
    import dopedefects_structure_properties as structure_properties
import sys
import tables

def init_pandas(data_dir, csvs):
    """
    Initiate the external properties data frames.

    Inputs
    ------
    data_dir :    The path to the data directory where the csvs are
    csvs     :    The name of the csv files where the extra properties
                  will be read in from.

    Outputs
    -------
    data     :    The pandas dataframe containing the extra properties
                  contained within the csv files passed into csvs
    """
    data = pandas.DataFrame()
    for csv in csvs:
        data = data.append(pandas.read_csv(data_dir + '/' + csv), sort=False, \
            ignore_index=True)
    return data

def clean_pandas(data):
    """
    Remove from consideration those locations in the pandas dataframe
    where there is not a POSCAR entry

    Inputs
    ------
    data  :   The pandas dataframe containing the atomic and structural
              properties

    Outputs
    -------
    data  :   The cleaned pandas dataframe contining the atomic and
              structural properties
    """
    entries = np.where(pandas.isnull(data['Bonds']) & pandas.isnull(\
        data['Angles']) & pandas.isnull(data['Coulomb']))[0]
    return data.drop(entries)

def append_atomic_properties(data, file_in="Elemental_properties.csv"):
    """
    Append the Elemental properties from the given csv file to the
    provided pandas database

    Inputs
    ------
    data    : The pandas dataframe containing the atomic and structural
              properteis
    file_in : The csv file containing the extra columns to append

    Outputs
    -------
    data    : The pandas dataframe containing the additional rows from
              file_in
    """
    atomic_properties = pandas.read_csv(file_in)
    for i in data.index.values:
        atom = data.at[i, "M"]
        try:
            entry = atomic_properties.loc[(atomic_properties['Symbol'] == \
                atom)].index.values[0]
        except:
            print("Atom '%s' not within the atomic_properties given, please \
fill in missing values to %s, or correct input data." %(atom, file_in))
            continue
        for column in atomic_properties.columns.values:
            data.at[i, column] = atomic_properties.at[entry, column]
    return data

def build_pandas(data_dir, csvs):
    """
    Builds the data matrix given the list of csvs and calculates the
    structural properties and builds the pandas dataframe containing
    all the given information.

    Inputs
    ------
    data_dir  : The directory containing the POSCAR files for the
                structures of interest
    csvs      : The file containing the csv files for adding additional
                properties to pandas dataframe

    Outputs
    -------
    data      : The pandas dataframe containing relevant information 
                for the ML processes.
    """
    data = init_pandas(data_dir, csvs)
    poscar_list = structure_extract.find_files(data_dir)
    ##Predefine elements for pandas as are matricies instead of single
    ##values.
    data["Bonds"] = None
    data["Bonds"] = data["Bonds"].astype(object)
    data["Angles"] = None
    data["Angles"] = data["Angles"].astype(object)
    data["Coulomb"] = None
    data["Coulomb"] = data["Coulomb"].astype(object)
    ##
    for poscar in poscar_list:
        crystal = structure_extract.id_crystal(poscar)
        dopant = structure_extract.impurity_type(poscar)
        site = structure_extract.dopant_site(poscar)
        if dopant == 'pure':
            dopant = site.replace("M_", "")
        try:
            entry = data.loc[(data['CdX'] == crystal) & (data['M'] == dopant) &\
                (data['Doping Site'] == site)].index[0]
        except:
            print("Cannot open dataframe entry for crystal %s, site %s, and \
dopant %s.  Please check .csv for %s" %(crystal, site, dopant, poscar))
            continue
        try:
            bonds_angles = structure_extract.geometry_defect(7, dopant, poscar)
            bonds = bonds_angles[0]
            angles = bonds_angles[1]
        except:
            print("Error (",sys.exc_info()[0], ")  determining bonds and angles\
for crystal %s, site %s and dopant %s.  Please check POSCAR"\
%(crystal, site, dopant))
            continue
        data.at[entry, "Bonds"] = bonds
        data.at[entry, "Angles"] = angles
        data.at[entry, "Coulomb"] = structure_properties.coulomb(bonds, \
            structure_extract.geometry_defect(0, dopant, poscar)[0])
    data = clean_pandas(data)
    data = append_atomic_properties(data)
    return data

def save_pandas(data, file_name):
    """
    Saves the given pandas dataframe into an hdf5 file to resume when
    needed

    Inputs
    ------
    data      : The pandas dataframe to save to file
    file_name : The name for the saved file

    Outputs
    -------
    N/A
    """
    data.to_hdf(file_name, 'data', data_columns=True)
    return

def init_data(file_name, data_dir=None, csvs=None, refresh=False, dna = True):
    """
    Calls the constructing functions or resumes from file to return the
    pandas dataframe containing relevant information for ML application

    Inputs
    ------
    file_name : The name for the data file to resume or save from or to
    data_dir  : The directory containing the POSCAR files of interest
    csvs      : A list containing csvs with additinal rows to append to
                the returned dataframe
    refresh   : Whether to force the generation of the pandas dataframe
                anew or allow for the reading in from file
    dna       : Remove any rows from the resulting dataframe with N/A
                entries (incomplete or non-calculated structures)

    Outputs
    -------
    data      : The cleaned (if requested) pandas dataframe containing
                the structual and atomic properties for use in ML.
    """
    if os.path.isfile(file_name):
        if not refresh:
            return pandas.read_hdf(file_name, 'data')
        elif data_dir and csvs:
            data = build_pandas(data_dir, csvs)
            if dna:
                data = data.dropna()
            save_pandas(data, file_name)
            return data
        else:
            raise Exception("Need directory and orginal *.csv files to\
parse into a file.")
    else:
        if data_dir and csvs:
            data = build_pandas(data_dir, csvs)
            if dna:
                data = data.dropna()
            save_pandas(data, file_name)
            return data
        else:
            raise Exception("Need directory and orginal *.csv files to\
parse into a file.")
