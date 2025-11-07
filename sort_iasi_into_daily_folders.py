import os
import sys
import glob
import shutil
import pdb
import datetime as dt


"""
	Script to sort IASI data downloaded via eumdac into daily folders and to separate the different
	METOP satellites (A, B, C) from each other. This will hopefully reduce the execution time of
	manage_iasi.py.
	- find files
	- identify date and METOP satellite for each file
	- sort into correct folder
"""


# paths:
path_in = "/mnt/d/heavy_data/IASI/new/"
path_out = "/mnt/d/heavy_data/IASI/mosaic/new/"


# find files:
files = sorted(glob.glob(path_in + "*.nc"))


# loop over files:
for i_f, file in enumerate(files):

	filename = os.path.basename(file)
	print(f"\rSorting data: {filename}", end='')

	# identify date:
	str_0 = "EUMP_"
	n_str_0 = len(str_0)
	i_str_0 = filename.find(str_0)
	date = filename[i_str_0+n_str_0:i_str_0+n_str_0+8]
	try:
		date_dt = dt.datetime.strptime(date, "%Y%m%d")
	except:
		print("Could not recognize date....")
		pdb.set_trace()

	# identify METOP satellite:
	str_1 = ",METOP"
	n_str_1 = len(str_1)
	i_str_1 = filename.find(str_1)
	metop = filename[i_str_1+n_str_1:i_str_1+n_str_1+1]

	# create folder to save file to if not existing:
	new_dir_name = f"{date}_{metop}"
	path_out_f = path_out + new_dir_name + "/"
	new_dir_dir = os.path.dirname(path_out_f)
	if not os.path.exists(new_dir_dir):
		os.makedirs(new_dir_dir)

	# move file:
	shutil.move(file, path_out_f + filename)


print("")
