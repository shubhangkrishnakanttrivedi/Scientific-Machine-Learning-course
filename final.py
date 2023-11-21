#imports
import math
import numpy as np
import os
import glob
from numpy import array
import pandas as pd
import re
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import zipfile
from pathlib import Path
from scipy.spatial import KDTree
from torch.nn.functional import normalize
from torch.optim import Adam
import time


# #  Print options

torch.set_printoptions(profile="full")
torch.set_printoptions(precision=10)
#torch.set_printoptions(linewidth=200)
torch.set_default_tensor_type(torch.FloatTensor)
# #  PyTorch Version
print(f"PyTorch Version: ", torch.__version__)

print(f" ")

# #  Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using :  {device}")
print(f" ")



# #  Access to dataset (training and test)

data_dir_4_atoms = Path("/storage/home/hhive1/strivedi44/scratch/SPARC/PyTorch/Combined_input_files_AL_4_ATOMS_final_parameters_with_output/Combined_input_files_AL_4_ATOMS_final_parameters_with_output")  # 4 atoms dataset for training
data_dir_32_atoms = Path("/storage/home/hhive1/strivedi44/scratch/SPARC/PyTorch/consolidated_input_files_32_atoms/consolidated_input_files_32_atoms")              # 32 atoms dataset for training
data_dir_108_atoms = Path("/storage/home/hhive1/strivedi44/scratch/SPARC/PyTorch/FFinal_Test_set_Output_108_Atoms_With_A_Vacancy/FFinal_Test_set_Output_108_Atoms_With_A_Vacancy")             # 108 atoms dataset for test

output_path = Path("/storage/home/hhive1/strivedi44/scratch/SPARC/PyTorch")    ## output files




# #  Checking if the folder directory is correct

if data_dir_4_atoms.is_dir():
        print(f"{data_dir_4_atoms} is in right place")
        print(f" ")

if data_dir_32_atoms.is_dir():
        print(f"{data_dir_32_atoms} is in right place")
        print(f" ")

if data_dir_108_atoms.is_dir():
        print(f"{data_dir_108_atoms} is in right place")
        print(f" ")



# #   Directories and file paths Block

ion_files_4_atoms = sorted(glob.glob(os.path.join(data_dir_4_atoms, "*.ion")))

ion_files_32_atoms = sorted(glob.glob(os.path.join(data_dir_32_atoms, "*.ion")))

ion_files_108_atoms =sorted(glob.glob(os.path.join(data_dir_108_atoms, "*.ion")))

txt_files_4_atoms =sorted(glob.glob(os.path.join(data_dir_4_atoms, "*.txt")))

txt_files_32_atoms =sorted(glob.glob(os.path.join(data_dir_32_atoms, "*.txt")))

txt_files_108_atoms = sorted(glob.glob(os.path.join(data_dir_108_atoms, "*.txt")))

out_files_4_atoms = sorted(glob.glob(os.path.join(data_dir_4_atoms, "*.out")))

out_files_32_atoms = sorted(glob.glob(os.path.join(data_dir_32_atoms, "*.out")))

out_files_108_atoms = sorted(glob.glob(os.path.join(data_dir_108_atoms, "*.out")))

output_files_108_atoms = sorted(glob.glob(os.path.join(output_path, "*.txt")))





# #  Input parameters Block

cell_4 = 7.653391;                 # Cell_size_of_system_containing_4_atoms

cell_32 = 15.265387350652707;       # Cell_size_of_system_containing_32_atoms

cell_108 = 22.96017822958;           # Cell_size_of_system_containing_108_atoms


NX_4 = 40;                         # No. of Grid points in cell_1_along X direction
NY_4 = 40;                         # No. of Grid points in cell_1_along Y direction
NZ_4 = 40;                         # No. of Grid points in Cell_1_along Z direction


NX_32 = 80;
NY_32 = 80;
NZ_32 = 80;


NX_108 = 110;
NY_108 = 110;
NZ_108 = 110;
  

N_atoms_4 = 4;
N_atoms_32 = 32;
N_atoms_108 = 107; 


N_descriptors = 4;    # No. of descriptors to capture the information

N_dim = 3;

# These below parameters might be same for all data

K_dist_4 = 4;     ## No. of smallest distance values for each grid point
K_pos_4 = 2;     ## No. of smallest/nearest atom considered from each atom

K_dist_32 = 20;
K_pos_32 = 2;

K_dist_108 = 20;
K_pos_108 = 2;


rad_cut_dist_4 = 40;   ## cuttof radius for sparse tensors

rad_cut_dist_32 = 40;

rad_cut_dist_108 = 40;





## Defining Grid


## 4 atoms
grid_spacing_X_4_atoms = torch.linspace(0, cell_4, NX_4)
grid_spacing_Y_4_atoms = torch.linspace(0, cell_4, NY_4)
grid_spacing_Z_4_atoms = torch.linspace(0, cell_4, NZ_4)

grid_spacing_X_4_atoms = grid_spacing_X_4_atoms.view(NX_4,1,1,1).expand(*(NX_4, NX_4, NX_4, 1)).contiguous()
grid_spacing_Y_4_atoms = grid_spacing_Y_4_atoms.view(1,NY_4,1,1).expand(*(NY_4, NY_4, NY_4, 1)).contiguous()
grid_spacing_Z_4_atoms = grid_spacing_Z_4_atoms.view(1,1,NZ_4,1).expand(*(NZ_4, NZ_4,NZ_4, 1)).contiguous()
catee_4 = torch.cat((grid_spacing_X_4_atoms, grid_spacing_Y_4_atoms, grid_spacing_Z_4_atoms), dim = 3)
catee_4_red = catee_4.view(NX_4*NY_4*NZ_4,N_dim)
expanded_catee_4_red = (catee_4_red.repeat(1, N_atoms_4)).expand(NX_4*NY_4*NZ_4, N_dim*N_atoms_4)
kd_tree_catee_4_red = KDTree(catee_4_red)


### 32 atoms
grid_spacing_X_32_atoms = torch.linspace(0, cell_32, NX_32)
grid_spacing_Y_32_atoms = torch.linspace(0, cell_32, NY_32)
grid_spacing_Z_32_atoms = torch.linspace(0, cell_32, NZ_32)

grid_spacing_X_32_atoms = grid_spacing_X_32_atoms.view(NX_32,1,1,1).expand(*(NX_32, NX_32, NX_32, 1)).contiguous()
grid_spacing_Y_32_atoms = grid_spacing_Y_32_atoms.view(1,NY_32,1,1).expand(*(NY_32, NY_32, NY_32, 1)).contiguous()
grid_spacing_Z_32_atoms = grid_spacing_Z_32_atoms.view(1,1,NZ_32,1).expand(*(NZ_32, NZ_32,NZ_32, 1)).contiguous()
catee_32 = torch.cat((grid_spacing_X_32_atoms, grid_spacing_Y_32_atoms, grid_spacing_Z_32_atoms), dim = 3)
catee_32_red = catee_32.view(NX_32*NY_32*NZ_32,N_dim)
expanded_catee_32_red = (catee_32_red.repeat(1, N_atoms_32)).expand(NX_32*NY_32*NZ_32, N_dim*N_atoms_32)
kd_tree_catee_32_red = KDTree(catee_32_red)


### 107 atoms
grid_spacing_X_108_atoms = torch.linspace(0, cell_108, NX_108)
grid_spacing_Y_108_atoms = torch.linspace(0, cell_108, NY_108)
grid_spacing_Z_108_atoms = torch.linspace(0, cell_108, NZ_108)

grid_spacing_X_108_atoms = grid_spacing_X_108_atoms.view(NX_108,1,1,1).expand(*(NX_108, NX_108, NX_108, 1)).contiguous()
grid_spacing_Y_108_atoms = grid_spacing_Y_108_atoms.view(1,NY_108,1,1).expand(*(NY_108, NY_108, NY_108, 1)).contiguous()
grid_spacing_Z_108_atoms = grid_spacing_Z_108_atoms.view(1,1,NZ_108,1).expand(*(NZ_108, NZ_108,NZ_108, 1)).contiguous()
catee_108 = torch.cat((grid_spacing_X_108_atoms, grid_spacing_Y_108_atoms, grid_spacing_Z_108_atoms), dim = 3)
catee_108_red = catee_108.view(NX_108*NY_108*NZ_108,N_dim)
expanded_catee_108_red = (catee_108_red.repeat(1, N_atoms_108)).expand(NX_108*NY_108*NZ_108, N_dim*N_atoms_108)
kd_tree_catee_108_red = KDTree(catee_108_red)





## Dataset Block

####################### 4 ATOMS DATASET #########################################################################################33


class CustomDataset_4_atoms(Dataset):

        print(f"went inside Dataset loop")

        def __init__(self, filenames_having_4_atoms, position_files_of_4_atoms, electronic_information_files_of_4_atoms):

 #               print(f"went inside Def_init_loop")

                self.filenames_having_4_atoms = filenames_having_4_atoms
                self.position_files_of_4_atoms = position_files_of_4_atoms
                self.electronic_information_files_of_4_atoms = electronic_information_files_of_4_atoms
                #self.batch_size_of_4_atoms = batch_size_of_4_atoms
      #          print(type(self.filenames_having_4_atoms))
       #         print(type(self.position_files_of_4_atoms))

        def __len__(self):

#                print("went inside self_length_loop")
                return len(self.filenames_having_4_atoms)
                return len(self.position_files_of_4_atoms)
                return len(self.electronic_information_files_of_4_atoms)
       

        def __getitem__(self, idx):
                global catenated_data_of_4_atoms, pos_data_of_4_atoms, elec_data_of_4_atoms, package_4
 #               print(f"went inside getitem_loop")

                single_containing_4_atoms = self.filenames_having_4_atoms[(idx)*batch_size_4_atoms:(idx+1)*batch_size_4_atoms]

                single_position_of_4_atoms = self.position_files_of_4_atoms[(idx)*batch_size_4_atoms:(idx+1)*batch_size_4_atoms]

                single_electronic_information_files_of_4_atoms = self.electronic_information_files_of_4_atoms[(idx)*batch_size_4_atoms:(idx+1)*batch_size_4_atoms]



                ## Electron Densities
                
                u_cat_4 = 0;
                for f_4_atoms in single_containing_4_atoms:
                        df_4_atoms = pd.read_csv(f_4_atoms, header = None)
  #                      print(f"reading {f_4_atoms}")
                        data_of_4_atoms = torch.tensor(df_4_atoms.values)
                        grid_data_of_4_atoms = data_of_4_atoms.view(1,NX_4*NY_4*NZ_4,1)
                       
                        if u_cat_4 == 0:
                          catenated_data_of_4_atoms = torch.zeros(1,NX_4*NY_4*NZ_4,1)
                          catenated_data_of_4_atoms = torch.cat([catenated_data_of_4_atoms, grid_data_of_4_atoms], dim=0)
                          catenated_data_of_4_atoms = catenated_data_of_4_atoms[1:,:,:]
                          u_cat_4=u_cat_4+1;

                        else:
                          catenated_data_of_4_atoms = torch.cat([catenated_data_of_4_atoms, grid_data_of_4_atoms], dim=0)
                catenated_data_of_4_atoms = ((catenated_data_of_4_atoms.squeeze(0)).view(batch_size_4_atoms, NX_4*NY_4*NZ_4)).squeeze(0)
                catenated_data_of_4_atoms = (catenated_data_of_4_atoms.float()).squeeze(0)
               # catenated_data_of_4_atoms = catenated_data_of_4_atoms.               



 # print(catenated_data_of_4_atoms)
               # print(catenated_data_of_4_atoms.size())
               # print(catenated_data_of_4_atoms.dim())
             
                ## Atom positions
               
                u_package_4 = 0;
                u_pos_4 = 0;
                for f_4_atoms in single_position_of_4_atoms:
                        pos_4_atoms = pd.read_csv(f_4_atoms, header = None,  sep=r"\s\s+", skiprows=21, nrows = N_atoms_4)
 #                       print(f"reading {f_4_atoms}")
                        pos_tensor_of_4_atoms = torch.tensor(pos_4_atoms.values)
                        pos_mul_data_of_4_atoms = torch.mul(pos_tensor_of_4_atoms, cell_4)
                        pos_grid_data_of_4_atoms = pos_mul_data_of_4_atoms.view(1, N_atoms_4, 3)
                       # print(pos_grid_data_of_4_atoms, pos_grid_data_of_4_atoms.dim(), pos_grid_data_of_4_atoms.size())                       
                        pos_grid_data_of_4_atoms = pos_grid_data_of_4_atoms.squeeze(0)
                       # print(pos_grid_data_of_4_atoms, pos_grid_data_of_4_atoms.dim(), pos_grid_data_of_4_atoms.size())
  ## Finding Vectors   (Main)  ######################################################################################################################################################################

                        pos_4_atoms_transformed_view = pos_grid_data_of_4_atoms.reshape(1, N_atoms_4*N_dim)
                    #    print(pos_4_atoms_transformed_view, pos_4_atoms_transformed_view.size(), pos_4_atoms_transformed_view.dim())
                        expanded_pos_4_atom_transformed_view = pos_4_atoms_transformed_view.expand(NX_4*NY_4*NZ_4, N_atoms_4*N_dim)
                     #   print(expanded_pos_4_atom_transformed_view, expanded_pos_4_atom_transformed_view.size(), expanded_pos_4_atom_transformed_view.dim())
                        subtracted_tensors = torch.sub(expanded_pos_4_atom_transformed_view, expanded_catee_4_red)
                      #  print(subtracted_tensors, subtracted_tensors.size(), subtracted_tensors.dim())
                        subtracted_tensor_long_view = subtracted_tensors.view(N_atoms_4*NX_4*NY_4*NZ_4, N_dim)
                        normalized_subtracted_tensor = normalize(subtracted_tensor_long_view, p=2, dim = 1) 
#print(normalized_subtracted_tensor)   ## Normalized unit vector from each grid point
                       
                       
                      
               				## KDTree method 
                
                       
                        kd_tree_pos_4 = KDTree(pos_grid_data_of_4_atoms)
                    
                        sparse_matrix_distance_4 = kd_tree_catee_4_red.sparse_distance_matrix(kd_tree_pos_4, rad_cut_dist_4)  ## Distance matrix with sparseness
                
                        x_4=sparse_matrix_distance_4.toarray()
  
                        sparse_tensor_4 = torch.from_numpy(x_4)
                
                        # print(sparse_matrix_distance_4)
                

                        sparse_matrix_itself_4 = kd_tree_pos_4.sparse_distance_matrix(kd_tree_pos_4, rad_cut_dist_4)      
                
                        x_4_itself = sparse_matrix_itself_4.toarray()
                        sparse_tensor_itself_4 = torch.from_numpy(x_4_itself)
                
                       
                         #################### TOPK's are Unsorted,   try later with sorted  ##############################################################################################################################################

                        val_s_t_4,ind_s_t_4 = torch.topk(sparse_tensor_4, K_dist_4, largest = False, sorted = True)
                        # print(val_s_t_4)
                        val_s_t_i_4, ind_s_t_i_4 = torch.topk(sparse_tensor_itself_4, K_pos_4, largest = False, sorted = True)
                        #print(val_s_t_i_4, ind_s_t_i_4)
                 
                        indices_tensor = ind_s_t_4.view(1, NX_4*NY_4*NZ_4*K_dist_4)
                        list_indices_tensor = indices_tensor.tolist()  
                        # print(indices_tensor_1_st)
                        # flatten_indices_tensor_1_st = torch.flatten(indices_tensor_1_st) 
                
                        ind_s_t_i_4_list = ind_s_t_i_4[list_indices_tensor]
                        ind_s_t_i_4_rearranged_list = ind_s_t_i_4_list.view(NX_4*NY_4*NZ_4, K_dist_4*K_pos_4)              

             
                ## Done without weights in one line
 
                        added_tensor = torch.add(ind_s_t_i_4_rearranged_list, torch.transpose(((torch.arange(0, NX_4*NY_4*NZ_4, 1)).expand(K_pos_4*K_dist_4, NX_4*NY_4*NZ_4)), 0, 1), alpha=N_atoms_4)
                        added_tensor_changed_view =  added_tensor.view(1, NX_4*NY_4*NZ_4*K_pos_4*K_dist_4)
                        added_tensor_changed_view_list = added_tensor_changed_view.tolist()
                        selecting_vectors = normalized_subtracted_tensor[added_tensor_changed_view_list]
                        selecting_vectors_rearranged = selecting_vectors.view(NX_4*NY_4*NZ_4, K_pos_4*N_dim*K_dist_4)
                        selecting_vectors_compartmentalized = torch.transpose(((torch.transpose(selecting_vectors_rearranged, 0, 1)).view(K_dist_4, K_pos_4*N_dim, NX_4*NY_4*NZ_4)), 1, 2)
                        # print(selecting_vectors_compartmentalized)
                      
 
                ## When selecting more K_pos_4 values , change the below line and make partitions and permutations so that all the dot products are taken into account##########################################################
              
                
               ###################### torch acos consumes double memory, it is better to keep in terms of cos   ###################################################################
                 
                        selecting_vectors_dot = torch.acos(torch.sum((((selecting_vectors_compartmentalized[:,:,:N_dim]).reshape(NX_4*NY_4*NZ_4*N_dim*K_dist_4, 1))*((selecting_vectors_compartmentalized[:,:,N_dim:]).reshape(NX_4*NY_4*NZ_4*N_dim*K_dist_4, 1))).reshape(K_dist_4, NX_4*NY_4*NZ_4,N_dim), dim = 2))
                        # print(selecting_vectors_dot)
                      
                        selecting_vectors_dot_grid_wise = torch.transpose(selecting_vectors_dot, 0, 1) 
                        # print(selecting_vectors_dot_grid_wise)
                        
                       # dot_tensor_phi = ((torch.tensor([0,1,0])).expand(NX_4*NY_4*NZ_4*K_dist_4,N_dim)).view(NX_4*NY_4*NZ_4*N_dim*K_dist_4, 1)
                       # selecting_vectors_dot_phi = torch.acos(torch.sum((((selecting_vectors_compartmentalized[:,:,:N_dim]).reshape(NX_4*NY_4*NZ_4*N_dim*K_dist_4, 1))*(((torch.tensor([0,1,0])).expand(NX_4*NY_4*NZ_4*K_dist_4,N_dim)).view(NX_4*NY_4*NZ_4*N_dim*K_dist_4, 1))).reshape(K_dist_4, NX_4*NY_4*NZ_4,N_dim), dim = 2))

                        ## 5D tensor

                        package_4_inter = torch.cat([val_s_t_4, selecting_vectors_dot_grid_wise], dim =1) 
                        # print(package_4)
                 
                        
                        if u_package_4 == 0 and u_pos_4 == 0:
                          package_4 = torch.zeros(1, K_dist_4+K_dist_4*(K_pos_4-1))
                          package_4 = torch.cat([package_4, package_4_inter], dim = 0)
                          package_4 = package_4[1:,:]
                          u_package_4 = u_package_4+1;

                          pos_data_of_4_atoms = torch.zeros(N_atoms_4, N_dim)
                          pos_data_of_4_atoms = torch.cat([pos_data_of_4_atoms, pos_grid_data_of_4_atoms], dim=0)
                          pos_data_of_4_atoms = pos_data_of_4_atoms[N_atoms_4:,:]
                          u_pos_4=u_pos_4+1;

                        else:
                          package_4 = torch.cat([package_4, package_4_inter], dim = 0)
                          pos_data_of_4_atoms = torch.cat([pos_data_of_4_atoms, pos_grid_data_of_4_atoms], dim=0)
  
               # package_4 = (((torch.transpose(package_4, 0, 1)).view(batch_size_4_atoms, K_dist_4*K_pos_4, NX_4, NY_4, NZ_4)).float()).squeeze(0)   


                ## Atom Type
              
                u_elec_4=0; 
                for f_4_atoms in single_electronic_information_files_of_4_atoms:
                        elec_config_4_atoms = pd.read_csv(f_4_atoms, header = None, sep="\s+", skiprows=84, nrows = 1)
#                        print(f"reading {f_4_atoms}")
                        elec_config_4_atoms_reduced_int_form = elec_config_4_atoms.loc[0].iat[-1]
                        elec_config_4_atoms_tensor = torch.tensor(elec_config_4_atoms_reduced_int_form)
                        elec_config_4_atoms_tensor_actual = (elec_config_4_atoms_tensor.expand(1, N_atoms_4))

                        if u_elec_4 == 0:   
                          elec_data_of_4_atoms = torch.zeros(1, N_atoms_4)
                          elec_data_of_4_atoms = torch.cat([elec_data_of_4_atoms, elec_config_4_atoms_tensor_actual])
                          elec_data_of_4_atoms = elec_data_of_4_atoms[1:,:]
                          u_elec_4=u_elec_4+1;
                        
                        else:
                          elec_data_of_4_atoms = torch.cat([elec_data_of_4_atoms, elec_config_4_atoms_tensor_actual])
                        
                elec_data_of_4_atoms = elec_data_of_4_atoms.squeeze(0)         
                elec_data_of_4_atoms = (((elec_data_of_4_atoms.view(1, N_atoms_4)).float()).squeeze(0)).squeeze(0)                    

                if idx == self.__len__():
                        raise IndexError
                
               # print(elec_data_of_4_atoms)
   
                
                return catenated_data_of_4_atoms, package_4, elec_data_of_4_atoms





####################### 32 ATOMS DATASET #########################################################################################33


class CustomDataset_32_atoms(Dataset):

       # print(f"went inside Dataset loop")

        def __init__(self, filenames_having_32_atoms, position_files_of_32_atoms, electronic_information_files_of_32_atoms):

                #print(f"went inside Def_init_loop")

                self.filenames_having_32_atoms = filenames_having_32_atoms
                self.position_files_of_32_atoms = position_files_of_32_atoms
                self.electronic_information_files_of_32_atoms = electronic_information_files_of_32_atoms
                #self.batch_size_of_32_atoms = batch_size_of_32_atoms
      #          print(type(self.filenames_having_32_atoms))
       #         print(type(self.position_files_of_32_atoms))

        def __len__(self):

               # print("went inside self_length_loop")
                return len(self.filenames_having_32_atoms)
                return len(self.position_files_of_32_atoms)
                return len(self.electronic_information_files_of_32_atoms)
       

        def __getitem__(self, idx):
                global catenated_data_of_32_atoms, pos_data_of_32_atoms, elec_data_of_32_atoms, package_32
              #  print(f"went inside getitem_loop")

                single_containing_32_atoms = self.filenames_having_32_atoms[(idx)*batch_size_32_atoms:(idx+1)*batch_size_32_atoms]

                single_position_of_32_atoms = self.position_files_of_32_atoms[(idx)*batch_size_32_atoms:(idx+1)*batch_size_32_atoms]

                single_electronic_information_files_of_32_atoms = self.electronic_information_files_of_32_atoms[(idx)*batch_size_32_atoms:(idx+1)*batch_size_32_atoms]



                ## Electron Densities
                
                u_cat_32 = 0;
                for f_32_atoms in single_containing_32_atoms:
                        df_32_atoms = pd.read_csv(f_32_atoms, header = None)
                        #print(f"reading {f_32_atoms}")
                        data_of_32_atoms = torch.tensor(df_32_atoms.values)
                        grid_data_of_32_atoms = data_of_32_atoms.view(1,NX_32*NY_32*NZ_32,1)
                       
                        if u_cat_32 == 0:
                          catenated_data_of_32_atoms = torch.zeros(1,NX_32*NY_32*NZ_32,1)
                          catenated_data_of_32_atoms = torch.cat([catenated_data_of_32_atoms, grid_data_of_32_atoms], dim=0)
                          catenated_data_of_32_atoms = catenated_data_of_32_atoms[1:,:,:]
                          u_cat_32=u_cat_32+1;

                        else:
                          catenated_data_of_32_atoms = torch.cat([catenated_data_of_32_atoms, grid_data_of_32_atoms], dim=0)
                catenated_data_of_32_atoms = ((catenated_data_of_32_atoms.squeeze(0)).view(batch_size_32_atoms, NX_32*NY_32*NZ_32)).squeeze(0)
                catenated_data_of_32_atoms = (catenated_data_of_32_atoms.float()).squeeze(0)
               # catenated_data_of_32_atoms = catenated_data_of_32_atoms.               



 # print(catenated_data_of_32_atoms)
               # print(catenated_data_of_32_atoms.size())
               # print(catenated_data_of_32_atoms.dim())
             
                ## Atom positions
               
                u_package_32 = 0;
                u_pos_32 = 0;
                for f_32_atoms in single_position_of_32_atoms:
                        pos_32_atoms = pd.read_csv(f_32_atoms, header = None,  sep="\t", skiprows=21, nrows = N_atoms_32)
                       # print(f"reading {f_32_atoms}")
                        pos_tensor_of_32_atoms = torch.tensor(pos_32_atoms.values)
                        pos_mul_data_of_32_atoms = torch.mul(pos_tensor_of_32_atoms, cell_32)
                        pos_grid_data_of_32_atoms = pos_mul_data_of_32_atoms.view(1, N_atoms_32, 3)
                       # print(pos_grid_data_of_32_atoms, pos_grid_data_of_32_atoms.dim(), pos_grid_data_of_32_atoms.size())                       
                        pos_grid_data_of_32_atoms = pos_grid_data_of_32_atoms.squeeze(0)
                       # print(pos_grid_data_of_32_atoms, pos_grid_data_of_32_atoms.dim(), pos_grid_data_of_32_atoms.size())
  ## Finding Vectors   (Main)  ######################################################################################################################################################################

                        pos_32_atoms_transformed_view = pos_grid_data_of_32_atoms.reshape(1, N_atoms_32*N_dim)
                    #    print(pos_32_atoms_transformed_view, pos_32_atoms_transformed_view.size(), pos_32_atoms_transformed_view.dim())
                        expanded_pos_32_atom_transformed_view = pos_32_atoms_transformed_view.expand(NX_32*NY_32*NZ_32, N_atoms_32*N_dim)
                     #   print(expanded_pos_32_atom_transformed_view, expanded_pos_32_atom_transformed_view.size(), expanded_pos_32_atom_transformed_view.dim())
                        subtracted_tensors = torch.sub(expanded_pos_32_atom_transformed_view, expanded_catee_32_red)
                      #  print(subtracted_tensors, subtracted_tensors.size(), subtracted_tensors.dim())
                        subtracted_tensor_long_view = subtracted_tensors.view(N_atoms_32*NX_32*NY_32*NZ_32, N_dim)
                        normalized_subtracted_tensor = normalize(subtracted_tensor_long_view, p=2, dim = 1) 
#print(normalized_subtracted_tensor)   ## Normalized unit vector from each grid point
                       
                       
                      
               				## KDTree method 
                
                       
                        kd_tree_pos_32 = KDTree(pos_grid_data_of_32_atoms)
                    
                        sparse_matrix_distance_32 = kd_tree_catee_32_red.sparse_distance_matrix(kd_tree_pos_32, rad_cut_dist_32)  ## Distance matrix with sparseness
                
                        x_32=sparse_matrix_distance_32.toarray()
  
                        sparse_tensor_32 = torch.from_numpy(x_32)
                
                        # print(sparse_matrix_distance_32)
                

                        sparse_matrix_itself_32 = kd_tree_pos_32.sparse_distance_matrix(kd_tree_pos_32, rad_cut_dist_32)      
                
                        x_32_itself = sparse_matrix_itself_32.toarray()
                        sparse_tensor_itself_32 = torch.from_numpy(x_32_itself)
                
                       
                         #################### TOPK's are Unsorted,   try later with sorted  ##############################################################################################################################################

                        val_s_t_32,ind_s_t_32 = torch.topk(sparse_tensor_32, K_dist_32, largest = False, sorted = True)
                        # print(val_s_t_32)
                        val_s_t_i_32, ind_s_t_i_32 = torch.topk(sparse_tensor_itself_32, K_pos_32, largest = False, sorted = True)
                        #print(val_s_t_i_32, ind_s_t_i_32)
                 
                        indices_tensor = ind_s_t_32.view(1, NX_32*NY_32*NZ_32*K_dist_32)
                        list_indices_tensor = indices_tensor.tolist()  
                        # print(indices_tensor_1_st)
                        # flatten_indices_tensor_1_st = torch.flatten(indices_tensor_1_st) 
                
                        ind_s_t_i_32_list = ind_s_t_i_32[list_indices_tensor]
                        ind_s_t_i_32_rearranged_list = ind_s_t_i_32_list.view(NX_32*NY_32*NZ_32, K_dist_32*K_pos_32)              

             
                ## Done without weights in one line
 
                        added_tensor = torch.add(ind_s_t_i_32_rearranged_list, torch.transpose(((torch.arange(0, NX_32*NY_32*NZ_32, 1)).expand(K_pos_32*K_dist_32, NX_32*NY_32*NZ_32)), 0, 1), alpha=N_atoms_32)
                        added_tensor_changed_view =  added_tensor.view(1, NX_32*NY_32*NZ_32*K_pos_32*K_dist_32)
                        added_tensor_changed_view_list = added_tensor_changed_view.tolist()
                        selecting_vectors = normalized_subtracted_tensor[added_tensor_changed_view_list]
                        selecting_vectors_rearranged = selecting_vectors.view(NX_32*NY_32*NZ_32, K_pos_32*N_dim*K_dist_32)
                        selecting_vectors_compartmentalized = torch.transpose(((torch.transpose(selecting_vectors_rearranged, 0, 1)).view(K_dist_32, K_pos_32*N_dim, NX_32*NY_32*NZ_32)), 1, 2)
                        # print(selecting_vectors_compartmentalized)
                      
 
                ## When selecting more K_pos_32 values , change the below line and make partitions and permutations so that all the dot products are taken into account##########################################################
              
                
               ###################### torch acos consumes double memory, it is better to keep in terms of cos   ###################################################################
                 
                        selecting_vectors_dot = torch.acos(torch.sum((((selecting_vectors_compartmentalized[:,:,:N_dim]).reshape(NX_32*NY_32*NZ_32*N_dim*K_dist_32, 1))*((selecting_vectors_compartmentalized[:,:,N_dim:]).reshape(NX_32*NY_32*NZ_32*N_dim*K_dist_32, 1))).reshape(K_dist_32, NX_32*NY_32*NZ_32,N_dim), dim = 2))
                        # print(selecting_vectors_dot)
                      
                        selecting_vectors_dot_grid_wise = torch.transpose(selecting_vectors_dot, 0, 1) 
                        # print(selecting_vectors_dot_grid_wise)
                        
                       # dot_tensor_phi = ((torch.tensor([0,1,0])).expand(NX_32*NY_32*NZ_32*K_dist_32,N_dim)).view(NX_32*NY_32*NZ_32*N_dim*K_dist_32, 1)
                       # selecting_vectors_dot_phi = torch.acos(torch.sum((((selecting_vectors_compartmentalized[:,:,:N_dim]).reshape(NX_32*NY_32*NZ_32*N_dim*K_dist_32, 1))*(((torch.tensor([0,1,0])).expand(NX_32*NY_32*NZ_32*K_dist_32,N_dim)).view(NX_32*NY_32*NZ_32*N_dim*K_dist_32, 1))).reshape(K_dist_32, NX_32*NY_32*NZ_32,N_dim), dim = 2))

                        ## 5D tensor

                        package_32_inter = torch.cat([val_s_t_32, selecting_vectors_dot_grid_wise], dim =1) 
                        # print(package_32)
                 
                        
                        if u_package_32 == 0 and u_pos_32 == 0:
                          package_32 = torch.zeros(1, K_dist_32+K_dist_32*(K_pos_32-1))
                          package_32 = torch.cat([package_32, package_32_inter], dim = 0)
                          package_32 = package_32[1:,:]
                          u_package_32 = u_package_32+1;

                          pos_data_of_32_atoms = torch.zeros(N_atoms_32, N_dim)
                          pos_data_of_32_atoms = torch.cat([pos_data_of_32_atoms, pos_grid_data_of_32_atoms], dim=0)
                          pos_data_of_32_atoms = pos_data_of_32_atoms[N_atoms_32:,:]
                          u_pos_32=u_pos_32+1;

                        else:
                          package_32 = torch.cat([package_32, package_32_inter], dim = 0)
                          pos_data_of_32_atoms = torch.cat([pos_data_of_32_atoms, pos_grid_data_of_32_atoms], dim=0)
  
               # package_32 = (((torch.transpose(package_32, 0, 1)).view(batch_size_32_atoms, K_dist_32*K_pos_32, NX_32, NY_32, NZ_32)).float()).squeeze(0)   


                ## Atom Type
              
                u_elec_32=0; 
                for f_32_atoms in single_electronic_information_files_of_32_atoms:
                        elec_config_32_atoms = pd.read_csv(f_32_atoms, header = None, sep="\s+", skiprows=84, nrows = 1)
                        print(f"reading {f_32_atoms}")
                        elec_config_32_atoms_reduced_int_form = elec_config_32_atoms.loc[0].iat[-1]
                        elec_config_32_atoms_tensor = torch.tensor(elec_config_32_atoms_reduced_int_form)
                        elec_config_32_atoms_tensor_actual = (elec_config_32_atoms_tensor.expand(1, N_atoms_32))

                        if u_elec_32 == 0:   
                          elec_data_of_32_atoms = torch.zeros(1, N_atoms_32)
                          elec_data_of_32_atoms = torch.cat([elec_data_of_32_atoms, elec_config_32_atoms_tensor_actual])
                          elec_data_of_32_atoms = elec_data_of_32_atoms[1:,:]
                          u_elec_32=u_elec_32+1;
                        
                        else:
                          elec_data_of_32_atoms = torch.cat([elec_data_of_32_atoms, elec_config_32_atoms_tensor_actual])
                        
                elec_data_of_32_atoms = elec_data_of_32_atoms.squeeze(0)         
                elec_data_of_32_atoms = (((elec_data_of_32_atoms.view(1, N_atoms_32)).float()).squeeze(0)).squeeze(0)                    

                if idx == self.__len__():
                        raise IndexError
                
               # print(elec_data_of_32_atoms)
   
                
                return catenated_data_of_32_atoms, package_32, elec_data_of_32_atoms







####################### 108 ATOMS DATASET #########################################################################################33


class CustomDataset_108_atoms(Dataset):

       # print(f"went inside Dataset loop")

        def __init__(self, filenames_having_108_atoms, position_files_of_108_atoms, electronic_information_files_of_108_atoms):

               # print(f"went inside Def_init_loop")

                self.filenames_having_108_atoms = filenames_having_108_atoms
                self.position_files_of_108_atoms = position_files_of_108_atoms
                self.electronic_information_files_of_108_atoms = electronic_information_files_of_108_atoms
                #self.batch_size_of_108_atoms = batch_size_of_108_atoms
      #          print(type(self.filenames_having_108_atoms))
       #         print(type(self.position_files_of_108_atoms))

        def __len__(self):

              #  print("went inside self_length_loop")
                return len(self.filenames_having_108_atoms)
                return len(self.position_files_of_108_atoms)
                return len(self.electronic_information_files_of_108_atoms)
       

        def __getitem__(self, idx):
                global catenated_data_of_108_atoms, pos_data_of_108_atoms, elec_data_of_108_atoms, package_108
             #   print(f"went inside getitem_loop")

                single_containing_108_atoms = self.filenames_having_108_atoms[(idx)*batch_size_108_atoms:(idx+1)*batch_size_108_atoms]

                single_position_of_108_atoms = self.position_files_of_108_atoms[(idx)*batch_size_108_atoms:(idx+1)*batch_size_108_atoms]

                single_electronic_information_files_of_108_atoms = self.electronic_information_files_of_108_atoms[(idx)*batch_size_108_atoms:(idx+1)*batch_size_108_atoms]



                ## Electron Densities
                
                u_cat_108 = 0;
                for f_108_atoms in single_containing_108_atoms:
                        df_108_atoms = pd.read_csv(f_108_atoms, header = None)
                       # print(f"reading {f_108_atoms}")
                        data_of_108_atoms = torch.tensor(df_108_atoms.values)
                        grid_data_of_108_atoms = data_of_108_atoms.view(1,NX_108*NY_108*NZ_108,1)
                       
                        if u_cat_108 == 0:
                          catenated_data_of_108_atoms = torch.zeros(1,NX_108*NY_108*NZ_108,1)
                          catenated_data_of_108_atoms = torch.cat([catenated_data_of_108_atoms, grid_data_of_108_atoms], dim=0)
                          catenated_data_of_108_atoms = catenated_data_of_108_atoms[1:,:,:]
                          u_cat_108=u_cat_108+1;

                        else:
                          catenated_data_of_108_atoms = torch.cat([catenated_data_of_108_atoms, grid_data_of_108_atoms], dim=0)
                catenated_data_of_108_atoms = ((catenated_data_of_108_atoms.squeeze(0)).view(batch_size_108_atoms, NX_108*NY_108*NZ_108)).squeeze(0)
                catenated_data_of_108_atoms = (catenated_data_of_108_atoms.float()).squeeze(0)
               # catenated_data_of_108_atoms = catenated_data_of_108_atoms.               



 # print(catenated_data_of_108_atoms)
               # print(catenated_data_of_108_atoms.size())
               # print(catenated_data_of_108_atoms.dim())
             
                ## Atom positions
               
                u_package_108 = 0;
                u_pos_108 = 0;
                for f_108_atoms in single_position_of_108_atoms:
                        pos_108_atoms = pd.read_csv(f_108_atoms, header = None,  sep="\t", skiprows=21, nrows = N_atoms_108)
                       # print(f"reading {f_108_atoms}")
                        pos_tensor_of_108_atoms = torch.tensor(pos_108_atoms.values)
                        pos_mul_data_of_108_atoms = torch.mul(pos_tensor_of_108_atoms, cell_108)
                        pos_grid_data_of_108_atoms = pos_mul_data_of_108_atoms.view(1, N_atoms_108, 3)
                       # print(pos_grid_data_of_108_atoms, pos_grid_data_of_108_atoms.dim(), pos_grid_data_of_108_atoms.size())                       
                        pos_grid_data_of_108_atoms = pos_grid_data_of_108_atoms.squeeze(0)
                       # print(pos_grid_data_of_108_atoms, pos_grid_data_of_108_atoms.dim(), pos_grid_data_of_108_atoms.size())
  ## Finding Vectors   (Main)  ######################################################################################################################################################################

                        pos_108_atoms_transformed_view = pos_grid_data_of_108_atoms.reshape(1, N_atoms_108*N_dim)
                    #    print(pos_108_atoms_transformed_view, pos_108_atoms_transformed_view.size(), pos_108_atoms_transformed_view.dim())
                        expanded_pos_108_atom_transformed_view = pos_108_atoms_transformed_view.expand(NX_108*NY_108*NZ_108, N_atoms_108*N_dim)
                     #   print(expanded_pos_108_atom_transformed_view, expanded_pos_108_atom_transformed_view.size(), expanded_pos_108_atom_transformed_view.dim())
                        subtracted_tensors = torch.sub(expanded_pos_108_atom_transformed_view, expanded_catee_108_red)
                      #  print(subtracted_tensors, subtracted_tensors.size(), subtracted_tensors.dim())
                        subtracted_tensor_long_view = subtracted_tensors.view(N_atoms_108*NX_108*NY_108*NZ_108, N_dim)
                        normalized_subtracted_tensor = normalize(subtracted_tensor_long_view, p=2, dim = 1) 
#print(normalized_subtracted_tensor)   ## Normalized unit vector from each grid point
                       
                       
                      
               				## KDTree method 
                
                       
                        kd_tree_pos_108 = KDTree(pos_grid_data_of_108_atoms)
                    
                        sparse_matrix_distance_108 = kd_tree_catee_108_red.sparse_distance_matrix(kd_tree_pos_108, rad_cut_dist_108)  ## Distance matrix with sparseness
                
                        x_108=sparse_matrix_distance_108.toarray()
  
                        sparse_tensor_108 = torch.from_numpy(x_108)
                
                        # print(sparse_matrix_distance_108)
                

                        sparse_matrix_itself_108 = kd_tree_pos_108.sparse_distance_matrix(kd_tree_pos_108, rad_cut_dist_108)      
                
                        x_108_itself = sparse_matrix_itself_108.toarray()
                        sparse_tensor_itself_108 = torch.from_numpy(x_108_itself)
                
                       
                         #################### TOPK's are Unsorted,   try later with sorted  ##############################################################################################################################################

                        val_s_t_108,ind_s_t_108 = torch.topk(sparse_tensor_108, K_dist_108, largest = False, sorted = True)
                        # print(val_s_t_108)
                        val_s_t_i_108, ind_s_t_i_108 = torch.topk(sparse_tensor_itself_108, K_pos_108, largest = False, sorted = True)
                        #print(val_s_t_i_108, ind_s_t_i_108)
                 
                        indices_tensor = ind_s_t_108.view(1, NX_108*NY_108*NZ_108*K_dist_108)
                        list_indices_tensor = indices_tensor.tolist()  
                        # print(indices_tensor_1_st)
                        # flatten_indices_tensor_1_st = torch.flatten(indices_tensor_1_st) 
                
                        ind_s_t_i_108_list = ind_s_t_i_108[list_indices_tensor]
                        ind_s_t_i_108_rearranged_list = ind_s_t_i_108_list.view(NX_108*NY_108*NZ_108, K_dist_108*K_pos_108)              

             
                ## Done without weights in one line
 
                        added_tensor = torch.add(ind_s_t_i_108_rearranged_list, torch.transpose(((torch.arange(0, NX_108*NY_108*NZ_108, 1)).expand(K_pos_108*K_dist_108, NX_108*NY_108*NZ_108)), 0, 1), alpha=N_atoms_108)
                        added_tensor_changed_view =  added_tensor.view(1, NX_108*NY_108*NZ_108*K_pos_108*K_dist_108)
                        added_tensor_changed_view_list = added_tensor_changed_view.tolist()
                        selecting_vectors = normalized_subtracted_tensor[added_tensor_changed_view_list]
                        selecting_vectors_rearranged = selecting_vectors.view(NX_108*NY_108*NZ_108, K_pos_108*N_dim*K_dist_108)
                        selecting_vectors_compartmentalized = torch.transpose(((torch.transpose(selecting_vectors_rearranged, 0, 1)).view(K_dist_108, K_pos_108*N_dim, NX_108*NY_108*NZ_108)), 1, 2)
                        # print(selecting_vectors_compartmentalized)
                      
 
                ## When selecting more K_pos_108 values , change the below line and make partitions and permutations so that all the dot products are taken into account##########################################################
              
                
               ###################### torch acos consumes double memory, it is better to keep in terms of cos   ###################################################################
                 
                        selecting_vectors_dot = torch.acos(torch.sum((((selecting_vectors_compartmentalized[:,:,:N_dim]).reshape(NX_108*NY_108*NZ_108*N_dim*K_dist_108, 1))*((selecting_vectors_compartmentalized[:,:,N_dim:]).reshape(NX_108*NY_108*NZ_108*N_dim*K_dist_108, 1))).reshape(K_dist_108, NX_108*NY_108*NZ_108,N_dim), dim = 2))
                        # print(selecting_vectors_dot)
                      
                        selecting_vectors_dot_grid_wise = torch.transpose(selecting_vectors_dot, 0, 1) 
                        # print(selecting_vectors_dot_grid_wise)
                        
                       # dot_tensor_phi = ((torch.tensor([0,1,0])).expand(NX_108*NY_108*NZ_108*K_dist_108,N_dim)).view(NX_108*NY_108*NZ_108*N_dim*K_dist_108, 1)
                       # selecting_vectors_dot_phi = torch.acos(torch.sum((((selecting_vectors_compartmentalized[:,:,:N_dim]).reshape(NX_108*NY_108*NZ_108*N_dim*K_dist_108, 1))*(((torch.tensor([0,1,0])).expand(NX_108*NY_108*NZ_108*K_dist_108,N_dim)).view(NX_108*NY_108*NZ_108*N_dim*K_dist_108, 1))).reshape(K_dist_108, NX_108*NY_108*NZ_108,N_dim), dim = 2))

                        ## 5D tensor

                        package_108_inter = torch.cat([val_s_t_108, selecting_vectors_dot_grid_wise], dim =1) 
                        # print(package_108)
                 
                        
                        if u_package_108 == 0 and u_pos_108 == 0:
                          package_108 = torch.zeros(1, K_dist_108+K_dist_108*(K_pos_108-1))
                          package_108 = torch.cat([package_108, package_108_inter], dim = 0)
                          package_108 = package_108[1:,:]
                          u_package_108 = u_package_108+1;

                          pos_data_of_108_atoms = torch.zeros(N_atoms_108, N_dim)
                          pos_data_of_108_atoms = torch.cat([pos_data_of_108_atoms, pos_grid_data_of_108_atoms], dim=0)
                          pos_data_of_108_atoms = pos_data_of_108_atoms[N_atoms_108:,:]
                          u_pos_108=u_pos_108+1;

                        else:
                          package_108 = torch.cat([package_108, package_108_inter], dim = 0)
                          pos_data_of_108_atoms = torch.cat([pos_data_of_108_atoms, pos_grid_data_of_108_atoms], dim=0)
  
               # package_108 = (((torch.transpose(package_108, 0, 1)).view(batch_size_108_atoms, K_dist_108*K_pos_108, NX_108, NY_108, NZ_108)).float()).squeeze(0)   


                ## Atom Type
              
                u_elec_108=0; 
                for f_108_atoms in single_electronic_information_files_of_108_atoms:
                        elec_config_108_atoms = pd.read_csv(f_108_atoms, header = None, sep="\s+", skiprows=84, nrows = 1)
                       # print(f"reading {f_108_atoms}")
                        elec_config_108_atoms_reduced_int_form = elec_config_108_atoms.loc[0].iat[-1]
                        elec_config_108_atoms_tensor = torch.tensor(elec_config_108_atoms_reduced_int_form)
                        elec_config_108_atoms_tensor_actual = (elec_config_108_atoms_tensor.expand(1, N_atoms_108))

                        if u_elec_108 == 0:   
                          elec_data_of_108_atoms = torch.zeros(1, N_atoms_108)
                          elec_data_of_108_atoms = torch.cat([elec_data_of_108_atoms, elec_config_108_atoms_tensor_actual])
                          elec_data_of_108_atoms = elec_data_of_108_atoms[1:,:]
                          u_elec_108=u_elec_108+1;
                        
                        else:
                          elec_data_of_108_atoms = torch.cat([elec_data_of_108_atoms, elec_config_108_atoms_tensor_actual])
                        
                elec_data_of_108_atoms = elec_data_of_108_atoms.squeeze(0)         
                elec_data_of_108_atoms = (((elec_data_of_108_atoms.view(1, N_atoms_108)).float()).squeeze(0)).squeeze(0)                    

                if idx == self.__len__():
                        raise IndexError
                
               # print(elec_data_of_108_atoms)
   
                
                return catenated_data_of_108_atoms, package_108, elec_data_of_108_atoms






## Dataset creation

## SET 1

batch_size_4_atoms = 1;

train_dataset_4_atoms = CustomDataset_4_atoms(filenames_having_4_atoms=txt_files_4_atoms, position_files_of_4_atoms = ion_files_4_atoms, electronic_information_files_of_4_atoms = out_files_4_atoms)


## SET 2

batch_size_32_atoms = 1;

train_dataset_32_atoms = CustomDataset_32_atoms(filenames_having_32_atoms=txt_files_32_atoms, position_files_of_32_atoms = ion_files_32_atoms, electronic_information_files_of_32_atoms = out_files_32_atoms)


## SET 3

batch_size_108_atoms = 1;

test_dataset_108_atoms = CustomDataset_108_atoms(filenames_having_108_atoms=txt_files_108_atoms, position_files_of_108_atoms = ion_files_108_atoms, electronic_information_files_of_108_atoms = out_files_108_atoms)



# # Dataloader Block 

train_loader_4_atoms = DataLoader(train_dataset_4_atoms, shuffle = True)

train_loader_32_atoms = DataLoader(train_dataset_32_atoms, shuffle = True)

test_loader_108_atoms = DataLoader(test_dataset_108_atoms, shuffle = False)


class ManualFunction_4(torch.autograd.Function):
        def forward(ctx,input_4):
           ctx.save_for_backward(input_4)
        # ctx.n = n   ## N_atoms*NX*NY*NZ in our case
           #print(package_4)
           #print(package_4.size())
           return ((torch.square(package_4))*3*K_dist_4).reshape(4000, 128)


        def backward(ctx, grad_output_4):
            input_4, = ctx.saved_tensors
            return grad_output_4

dtype = torch.float


## Parameters_4
w_4 = torch.rand((400,4000), device=device,dtype=dtype, requires_grad=True)
s_4 = torch.rand((1600,400), device=device,dtype=dtype, requires_grad=True)
c_4 = torch.rand((400,4000), device=device,dtype=dtype, requires_grad=True)
d_4 = torch.rand((500,1600), device=device,dtype=dtype, requires_grad=True)

learning_rate_4 = 0.0005


epochs_4 = 50


## Empty tensors for output

elec_data_4_out = torch.zeros(1)

elec_dens_4_out = torch.zeros(1, NX_4*NY_4*NZ_4)


empty_loss_4 = []
loss_count_4 = 0

## Training 

print("[INFO] training the 4_atoms_system network...")

startTime_4 = time.time()


for epoch_4 in range(0, epochs_4):

            for  i_4,  (catenated_data_of_4_atoms, package_4, elec_data_of_4_atoms) in enumerate(train_loader_4_atoms):
                
               
                catenated_data_of_4_atoms = catenated_data_of_4_atoms.to(device)

                package_4 = package_4.to(device)
 
                package_4 = torch.reciprocal(torch.add(((((torch.transpose(package_4.squeeze(0), 0, 1)).reshape(1, K_dist_4*NX_4*NY_4*NZ_4*K_pos_4))[:K_dist_4])),10)).float()

                elec_data_of_4_atoms = elec_data_of_4_atoms.to(device)
               

                Begin_4 = ManualFunction_4.apply
 
                elec_dens_4 =torch.matmul(d_4,torch.matmul(s_4,torch.matmul(w_4,Begin_4(package_4.float())).float()+torch.matmul(c_4,(((torch.transpose(package_4.float(), 0, 1)).reshape(1, 4*40*40*40*2)).reshape(4000,128)))))

                elec_dens_4 = (elec_dens_4.float()).view(batch_size_4_atoms, NX_4*NY_4*NZ_4)
                #print(f"reshape{elec_dens_4}")
                elec_dens_4 = (torch.div((elec_dens_4-torch.min(elec_dens_4)), (torch.max(elec_dens_4) - torch.min(elec_dens_4))))*10
                elec_dens_4 = (torch.div((elec_dens_4-torch.min(elec_dens_4)), (torch.max(elec_dens_4) - torch.min(elec_dens_4))))*10
                elec_dens_4 = elec_dens_4.exp()
               # print(f"exp{elec_dens_4}")
                elec_dens_4_sum = torch.sum(elec_dens_4, dim = 1)
               # print(f"sum{elec_dens_4_sum}")
                elec_dens_4_sum = elec_dens_4_sum.detach().numpy()[0]
              #  print(f"numpy{elec_dens_4_sum}")
                elec_dens_4 = elec_dens_4/elec_dens_4_sum*(torch.sum(elec_data_of_4_atoms, dim = 1))/((cell_4)/(NX_4)*(cell_4)/(NY_4)*(cell_4)/(NZ_4)) 
              #  print(f"final{elec_dens_4}")

              #  print(elec_dens_4, catenated_data_of_4_atoms)
                loss_4_para = torch.abs((elec_dens_4.squeeze(0)-catenated_data_of_4_atoms.squeeze(0))).sum()
                
              #  print(f"loss{loss_4_para}")     
                
                loss_4_para.backward()

                with torch.no_grad():
                        w_4 -= learning_rate_4 * w_4.grad
                        s_4 -= learning_rate_4 * s_4.grad
                        c_4 -= learning_rate_4 * c_4.grad
                        d_4 -= learning_rate_4 * d_4.grad

                        w_4.grad = None
                        s_4.grad = None
                        c_4.grad = None
                        d_4.grad = None       

                       
                ### for data purpose


                elec_data_4_out = torch.cat([elec_data_4_out, torch.sum(elec_dens_4, dim=1)*(cell_4)/(NX_4)*(cell_4)/(NY_4)*(cell_4)/(NZ_4)], dim = 0)  
              #  print(f"elec_Data_4 {elec_data_4_out}")
                
                elec_dens_4_out = torch.cat([elec_dens_4_out, elec_dens_4], dim = 0)          
              #  print(f"elec_Dens_4 {elec_dens_4_out}")
                
                loss_count_4 = loss_count_4+loss_4_para

                
            empty_loss_4.append(loss_count_4)
            print("[INFO] EPOCH_4: {}/{}".format(epoch_4 + 1, epochs_4), "  ", "loss {empty_loss_4[epoch_4]}") 
            print(f"loss at EPOCH :: {epoch_4} = {empty_loss_4[epoch_4]}")


endTime_4 = time.time()

print("[INFO] total time taken to train the 4_atoms_system : {:.2f}s".format(endTime_4 - startTime_4))


elec_data_4_out = elec_data_4_out[1:]

elec_dens_4_out = elec_dens_4_out[1:,:]

print(f"loss epoch_wise_4{empty_loss_4}")

            
elec_dens_4_out = elec_dens_4_out.view(epochs_4, len(list(out_files_4_atoms)), NX_4, NY_4, NZ_4)

elec_data_4_out = elec_data_4_out.view(epochs_4, len(list(out_files_4_atoms)))

#print(f" electrons_4 {elec_data_4_out}")





class ManualFunction_32(torch.autograd.Function):
        def forward(ctx,input_32):
           ctx.save_for_backward(input_32)
        # ctx.n = n   ## N_atoms*NX*NY*NZ in our case
           #print(package_32)
           #print(package_32.size())
           return ((torch.square(package_32))*3*K_dist_32).reshape(4000, 5120)


        def backward(ctx, grad_output_32):
            input_32, = ctx.saved_tensors
            return grad_output_32

dtype = torch.float



## Parameters_32
w_32 = w_4
s_32 = torch.rand((6400,400), device=device,dtype=dtype, requires_grad=True)
c_32 = c_4
d_32 = torch.rand((100,6400), device=device,dtype=dtype, requires_grad=True)


learning_rate_32 = 0.0005


epochs_32 = 20


## Empty tensors for output

elec_data_32_out = torch.zeros(1)

elec_dens_32_out = torch.zeros(1, NX_32*NY_32*NZ_32)


empty_loss_32 = []
loss_count_32 = 0

## Training 

print("[INFO] training the 32_atoms_system network...")

startTime_32 = time.time()


for epoch_32 in range(0, epochs_32):

            for  i_32,  (catenated_data_of_32_atoms, package_32, elec_data_of_32_atoms) in enumerate(train_loader_32_atoms):
                
               
                catenated_data_of_32_atoms = catenated_data_of_32_atoms.to(device)

                package_32 = package_32.to(device)
 
                package_32 = torch.reciprocal(torch.add(((((torch.transpose(package_32.squeeze(0), 0, 1)).reshape(1, K_dist_32*NX_32*NY_32*NZ_32*K_pos_32))[:K_dist_32])),10)).float()

                elec_data_of_32_atoms = elec_data_of_32_atoms.to(device)
               

                Begin_32 = ManualFunction_32.apply
 
                elec_dens_32 =torch.matmul(d_32,torch.matmul(s_32,torch.matmul(w_32,Begin_32(package_32.float())).float()+torch.matmul(c_32,(((torch.transpose(package_32.float(), 0, 1)).reshape(1, 20*80*80*80*2)).reshape(4000,5120)))))

                elec_dens_32 = (elec_dens_32.float()).view(batch_size_32_atoms, NX_32*NY_32*NZ_32)
                #print(f"reshape{elec_dens_32}")
                elec_dens_32 = (torch.div((elec_dens_32-torch.min(elec_dens_32)), (torch.max(elec_dens_32) - torch.min(elec_dens_32))))*10
                elec_dens_32 = (torch.div((elec_dens_32-torch.min(elec_dens_32)), (torch.max(elec_dens_32) - torch.min(elec_dens_32))))*10
                elec_dens_32 = elec_dens_32.exp()
               # print(f"exp{elec_dens_32}")
                elec_dens_32_sum = torch.sum(elec_dens_32, dim = 1)
               # print(f"sum{elec_dens_32_sum}")
                elec_dens_32_sum = elec_dens_32_sum.detach().numpy()[0]
              #  print(f"numpy{elec_dens_32_sum}")
                elec_dens_32 = elec_dens_32/elec_dens_32_sum*(torch.sum(elec_data_of_32_atoms, dim = 1))/((cell_32)/(NX_32)*(cell_32)/(NY_32)*(cell_32)/(NZ_32)) 
              #  print(f"final{elec_dens_32}")

              #  print(elec_dens_32, catenated_data_of_32_atoms)
                loss_32_para = torch.abs((elec_dens_32.squeeze(0)-catenated_data_of_32_atoms.squeeze(0))).sum()
                
              #  print(f"loss{loss_32_para}")     
                
                loss_32_para.backward()

                with torch.no_grad():
                        #w_32 -= learning_rate * w_32.grad
                        s_32 -= learning_rate_32 * s_32.grad
                        #c_32 -= learning_rate * c_32.grad
                        d_32 -= learning_rate_32 * d_32.grad

                        #w_32.grad = None
                        s_32.grad = None
                        #c_32.grad = None
                        d_32.grad = None       

                       
                ### for data purpose


                elec_data_32_out = torch.cat([elec_data_32_out, torch.sum(elec_dens_32, dim=1)*(cell_32)/(NX_32)*(cell_32)/(NY_32)*(cell_32)/(NZ_32)], dim = 0)  
              #  print(f"elec_Data_32 {elec_data_32_out}")
                
                elec_dens_32_out = torch.cat([elec_dens_32_out, elec_dens_32], dim = 0)          
              #  print(f"elec_Dens_32 {elec_dens_32_out}")
                
                loss_count_32 = loss_count_32+loss_32_para

                
            empty_loss_32.append(loss_count_32)
            print("[INFO] EPOCH_32: {}/{}".format(epoch_32 + 1, epochs_32))
            print(f"loss at EPOCH :: {epoch_32} = {empty_loss_32[epoch_32]}")


endTime_32 = time.time()

print("[INFO] total time taken to train the 32_atoms_system : {:.2f}s".format(endTime_32 - startTime_32))


elec_data_32_out = elec_data_32_out[1:]

elec_dens_32_out = elec_dens_32_out[1:,:]

print(f"loss epoch_wise_32{empty_loss_32}")

            
elec_dens_32_out = elec_dens_32_out.view(epochs_32, len(list(out_files_32_atoms)), NX_32, NY_32, NZ_32)

elec_data_32_out = elec_data_32_out.view(epochs_32, len(list(out_files_32_atoms)))

print(f"electrons_32 {elec_data_32_out}")




class ManualFunction_108(torch.autograd.Function):
        def forward(ctx,input_108):
           ctx.save_for_backward(input_108)
        # ctx.n = n   ## N_atoms*NX*NY*NZ in our case
          # print(package_108)
           #print(package_108.size())
           return ((torch.square(package_108))*3*K_dist_108).reshape(4000, 13310)


        def backward(ctx, grad_output_108):
            input_108, = ctx.saved_tensors
            return grad_output_108

dtype = torch.float





## Parameters_108

w_108 = w_32
s_108 = s_32
c_108 = c_32
d_108 = d_32


#learning_rate_108 = 0.000005


epochs_108 = 1


## Empty tensors for output

elec_data_108_out = torch.zeros(1)

elec_dens_108_out = torch.zeros(1, NX_108*NY_108*NZ_108)


empty_loss_108 = []
loss_count_108 = 0

## Training 

print("[INFO] testing the 108_atoms_system network...")

startTime_108 = time.time()


for epoch_108 in range(0, epochs_108):

            for  i_108,  (catenated_data_of_108_atoms, package_108, elec_data_of_108_atoms) in enumerate(test_loader_108_atoms):
                
               
                catenated_data_of_108_atoms = catenated_data_of_108_atoms.to(device)

                package_108 = package_108.to(device)
 
                package_108 = torch.reciprocal(torch.add(((((torch.transpose(package_108.squeeze(0), 0, 1)).reshape(1, K_dist_108*NX_108*NY_108*NZ_108*K_pos_108))[:K_dist_108])),10)).float()

                elec_data_of_108_atoms = elec_data_of_108_atoms.to(device)
               

                Begin_108 = ManualFunction_108.apply
 
                elec_dens_108 =torch.matmul(d_108,torch.matmul(s_108,torch.matmul(w_108,Begin_108(package_108.float())).float()+torch.matmul(c_108,(((torch.transpose(package_108.float(), 0, 1)).reshape(1, 20*110*110*110*2)).reshape(4000,13310)))))

                elec_dens_108 = (elec_dens_108.float()).view(batch_size_108_atoms, NX_108*NY_108*NZ_108)
                #print(f"reshape{elec_dens_108}")
                elec_dens_108 = (torch.div((elec_dens_108-torch.min(elec_dens_108)), (torch.max(elec_dens_108) - torch.min(elec_dens_108))))
                elec_dens_108 = (torch.div((elec_dens_108-torch.min(elec_dens_108)), (torch.max(elec_dens_108) - torch.min(elec_dens_108))))
                elec_dens_108 = elec_dens_108.exp()
               # print(f"exp{elec_dens_108}")
                elec_dens_108_sum = torch.sum(elec_dens_108, dim = 1)
               # print(f"sum{elec_dens_108_sum}")
                elec_dens_108_sum = elec_dens_108_sum.detach().numpy()[0]
              #  print(f"numpy{elec_dens_108_sum}")
                elec_dens_108 = elec_dens_108/elec_dens_108_sum*(torch.sum(elec_data_of_108_atoms, dim = 1))/((cell_108)/(NX_108)*(cell_108)/(NY_108)*(cell_108)/(NZ_108)) 
              #  print(f"final{elec_dens_108}")

              #  print(elec_dens_108, catenated_data_of_108_atoms)
                loss_108_para = torch.abs((elec_dens_108.squeeze(0)-catenated_data_of_108_atoms.squeeze(0))).sum()
                
              #  print(f"loss{loss_108_para}")     
                
                loss_108_para.backward()
         
                ### for data purpose


                elec_data_108_out = torch.cat([elec_data_108_out, torch.sum(elec_dens_108, dim=1)*(cell_108)/(NX_108)*(cell_108)/(NY_108)*(cell_108)/(NZ_108)], dim = 0)  
              #  print(f"elec_Data_108 {elec_data_108_out}")
                
                elec_dens_108_out = torch.cat([elec_dens_108_out, elec_dens_108], dim = 0)          
              #  print(f"elec_Dens_108 {elec_dens_108_out}")
                
                loss_count_108 = loss_108_para

                empty_loss_108.append(loss_count_108)
                print("[INFO] EPOCH_108: {}/{}".format(epoch_108 + 1, epochs_108), "  ", "loss {empty_loss_108[epoch_108]}") 
                print(f"loss at EPOCH :: {epoch_108} = {empty_loss_108[epoch_108]}")
           

endTime_108 = time.time()

print("[INFO] total time taken to test the 108_atoms_system : {:.2f}s".format(endTime_108 - startTime_108))


elec_data_108_out = elec_data_108_out[1:]

elec_dens_108_out = elec_dens_108_out[1:,:]

print(f"loss epoch_wise_108{empty_loss_108}")

            
elec_dens_108_out = elec_dens_108_out.view(epochs_108, len(list(out_files_108_atoms)), NX_108, NY_108, NZ_108)

elec_data_108_out = elec_data_108_out.view(epochs_108, len(list(out_files_108_atoms)))
print(f"electrons_108 {elec_data_108_out}")

print(f"{elec_dens_108_out}")


##OUTPUTS

## 1 vacancy

#elec_dens_108_1_vac = elec_dens_108_out[1,1,93,NY_108,NZ_108]
#elec_dens_108_1_vac = elec_dens_108_1_vac.view(1, NY_108*NZ_108)



#vac_1 = {'file_1':elec_dens_108_1_vac}
#torch.save(vac_1, output_path/'vac_1.txt')
#loaded_vac_1 = torch.load(output_path/'vac_1.txt')
#print(loaded_vac_1['file_1'] == elec_dens_108_1_vac)



## 2 vacancy

elec_dens_108_2_vac = elec_dens_108_out[0,1,2,:,:]
elec_dens_108_2_vac = elec_dens_108_2_vac.view(1, NY_108*NZ_108)



vac_2 = {'file_2':elec_dens_108_2_vac}
torch.save(vac_2,output_path/'vac_2.txt')
loaded_vac_2 = torch.load(output_path/'vac_2.txt')
print(loaded_vac_2['file_2'] == elec_dens_108_2_vac)




## 3 vacancy

elec_dens_108_3_vac = elec_dens_108_out[0,2,57,:,:]
elec_dens_108_3_vac = elec_dens_108_3_vac.view(1, NY_108*NZ_108)



vac_3 = {'file_3':elec_dens_108_3_vac}
torch.save(vac_3,output_path/'vac_3.txt')
loaded_vac_3 = torch.load(output_path/'vac_3.txt')
print(loaded_vac_3['file_3'] == elec_dens_108_3_vac)




## 4 vacancy

elec_dens_108_4_vac = elec_dens_108_out[0,3,3,:,:]
elec_dens_108_4_vac = elec_dens_108_4_vac.view(1, NY_108*NZ_108)



vac_4 = {'file_4':elec_dens_108_4_vac}
torch.save(vac_4, output_path/'vac_4.txt')
loaded_vac_4 = torch.load(output_path/'vac_4.txt')
print(loaded_vac_4['file_4'] == elec_dens_108_4_vac)




## 1 vacancy

elec_dens_108_5_vac = elec_dens_108_out[0,4,77,:,:]
elec_dens_108_5_vac = elec_dens_108_5_vac.view(1, NY_108*NZ_108)



vac_5 = {'file_5':elec_dens_108_5_vac}
torch.save(vac_5,output_path/'vac_5.txt')
loaded_vac_5 = torch.load(output_path/'vac_5.txt')
print(loaded_vac_5['file_5'] == elec_dens_108_5_vac)






print(f"able to reach till here")





