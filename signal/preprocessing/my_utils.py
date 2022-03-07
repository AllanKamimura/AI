import os
import pathlib
import scipy.io
import numpy as np

from PIL import Image

def read_matlab(matfile):
	"""
	Inputs:
		matfile [path]: path to the matlab file to read
	
	Outputs:
		matrix [Numpy.narray]: matrix in Python readable format
	"""

	try:
		m = scipy.io.loadmat(matfile)
		return m["data"]
	
	except Exception as e:
		m = np.loadtxt(matfile)
		return m

def read_csv(csvfile):
	"""
	Inputs:
		csvfile [path]: path to the csv file to read
	
	Outputs:
		matrix [Numpy.narray]: matrix in Python readable format
	"""
	return np.genfromtxt(csvfile, delimiter = ",")

def image_save(folder, mode):
	"""
	
	Read all files in a folder containing matrix in matlab format and save it as an .TIF image
	The .TIF images uses half-precision Float16
	Creates a new folder named "folder + _images"
	
	Inputs:
		folder [path]: path to the folder with matlab files to read
        mode [String]: file type, either "matlab" or "cvs"
	Outputs:
		None
	"""
	
	path = pathlib.PurePath(folder)
	parent_folder = path.parent
	folder_name = path.name
	new_folder = os.path.join(parent_folder, folder_name + "_images")
	
	try:
		os.mkdir(new_folder)
	except:
		pass

	file_list = os.listdir(folder)
	
	for file_name in file_list:
		this_path = os.path.join(folder, file_name)

		if os.path.isdir(this_path):
			continue
		
		if mode == "matlab":
			matrix = read_matlab(
				this_path
                )
   
		elif mode == "csv":
			matrix = read_csv(
				this_path
			)
			
		Image.fromarray(matrix).save(
				os.path.join(new_folder, file_name.split(".")[0] + ".tif")
			)
		
