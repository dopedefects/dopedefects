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
    """
    data = pandas.DataFrame()
    for csv in csvs:
        data = data.append(pandas.read_csv(data_dir + '/' + csv), sort=False, \
            ignore_index=True)
    return data

def clean_pandas(data):
    """
    """
    entries = np.where(pandas.isnull(data['Bonds']) & pandas.isnull(\
        data['Angles']) & pandas.isnull(data['Coulomb']))[0]
    return data.drop(entries)

def append_atomic_properties(data, file_in="Elemental_properties.csv"):
    """
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
            bonds, angles = structure_extract.geometry_defect(8, dopant, poscar)
        except:
            print("Error (",sys.exc_info()[0], ")  determining bonds and angles\
for crystal %s, site %s and dopant %s.  Please check POSCAR"\
%(crystal, site, dopant))
            continue
        data.at[entry, "Bonds"] = bonds
        data.at[entry, "Angles"] = angles
        data.at[entry, "Coulomb"] = structure_properties.coulomb(bonds, \
            structure_extract.geometry_defect(0, dopant, poscar))
    data = clean_pandas(data)
    data = append_atomic_properties(data)
    return data

def save_pandas(data, file_name):
    """
    """
    data.to_hdf(file_name, 'data', data_columns=True)
    return

def init_data(file_name, data_dir=None, csvs=None, refresh=False, dna = True):
    """
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
