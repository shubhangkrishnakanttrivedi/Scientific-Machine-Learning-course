# Scientific-Machine-Learning-course
Contains code and the SPARC atomic simulation files  

The simulation was performed for Aluminium bulk crystal having 4 atoms, 32 atoms and 108 atoms with a vacancy.
The 4 atoms and 32 atoms dataset for the simulation was retrieved from NOMAD database (https://nomad-lab.eu/prod/v1/gui/dataset/doi/10.17172/NOMAD/2021.06.07-1).
The dataset for 108 atoms (FCC cystal structure of AL) was manually created based on repeating the periodic arrangement of unit cell containing 4 atoms in 3D.
This dataset contains only atomic positions in a crystal.

To obtain electron density arrangement from the atomic position, the Density functional theory simulation was performed on SPARC software (https://github.com/SPARC-X/SPARC) while on Phoenix and Hive cluster of PACE at Georgia Tech.
The computing resources used are 600-900 processors on Dual Intel Xeon Gold 6226 CPUs @ 2.7 GHz having 192 GB RAM for atomic DFT simulation.

The code file : (final.py) was implemented on Pytorch Software. 
For running the ML simulation, the resources used were 72 processors on Xeon 6226 CPU @ 2.70GHz having 1.5 TB memory (Hive cluster of PACE at Georgia Tech).

Files:

In the GITHUB, 

Branch Name (shubhangkrishnakanttrivedi_4_atoms) contains : 

--- FFinal_Test_set_Output_108_Atoms_With_A_Vacancy/FFinal_Test_set_Output_108_Atoms_With_A_Vacancy  (FOLDER) having 5 Nos.  : output density, input and ion (atomic position and charge) files for 108 atom system with a vacancy 

--- consolidated_input_files_32_atoms/consolidated_input_files_32_atoms (FOLDER) having 10 Nos. : output density (.dens), (.txt), input (.inpt) and ion (.ion) (atomic position and charge) files for 32 atom system

--- The 4 atom system files are distrbuted in several folders named :

  1- Half_of_half
  2 - Half__of_other_half
  3 - other_half_of_half

--- And remaining 4 atom system files are in Branch Name (shubhangkrishnakanttrivedi_4_atoms_4), 
under the folder named : 

  4 - other_half_of_other_half


%%% The density file(.dens) or the text file (.txt) contains all the information of electron density at the GRID points. 
The format/convention is that (first all Z coordinates are spanned, while X and Y are zero, and then Y coordinate is increased to next consecutive grid and again all Z coordinate are spanned, and this happens until all Y and Z are spanned while X is still 0, and lastly, X is increased to next consecutive grid and again Y and Z are spanned in same order until all X is spanned).

**
The code has the name  (final.py)**
