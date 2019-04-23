### Main Use Case: 
Given a set of descriptors of an impurity, use a trained model to predict energy levels of the defect

### Details:
Analyze Best Descriptors from DFT calculations and use them to make predictions

#### Descriptors:
1. Period	Group	Site
2. Delta Ion. Rad.
3. Delta Cov. Rad.
4. Delta Ion. En.
5. Delta At. Rad.
6. Delta EA
7. Delta EN
8. Delta At. Num.	Delta Val.
9. Number of Cd Neighbors
10. Number of Te Neighbors
11. Corrected VBM (eV)
12. Corrected CBM (eV)

#### Others (to be explored e.g. atomic positions)
Properties of interest:
Electronic levels within the band gap
Defect formation enthalpies, ∆H

#### Program Specifics:
- Uses command line interface
- Should be python program
- Should be able to read in data from csv file(s).
- Should be able to read in Geometry from VASP in/output files.
- Should be able to reduce dimensionality of descriptors - intelligently select descriptors
- Should be able to save latent space for later use
- Should be able to reconstruct latent space when presented with new or additional data
- Should be able to be expanded to incorporate additional features (decoder/ predictive NNs)
- Should be able to predict properties based on latent space
- (soft goal) Model to be trained with Gaussian progress regression

![Usecases diagram](./usecases_diagram.png?raw=true)