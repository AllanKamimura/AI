import os
import pathlib
import scipy.io

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
		print(e)
		
def image_save_matlab(folder):
	"""
	
	Read all files in a folder containing matrix in matlab format and save it as an .TIF image
	The .TIF images uses half-precision Float16
	Creates a new folder named "folder + _images"
	
	Inputs:
		folder [path]: path to the folder with matlab files to read
	
	Outputs:
		None
	"""	
	
	path = pathlib.PurePath(folder)
	parent_folder = path.parent
	folder_name = path.name
	new_folder = os.path.join(parent_folder, folder_name + "_images")
	
	os.mkdir(new_folder)
	
	file_list = os.listdir(folder)
	
	for file_name in file_list:
		matrix = read_matlab(file_)
		Image.fromarray(matrix).save(
                os.path.join(new_folder, file_name.split(".")[0] + ".tif")
            )
		
