import numpy as np
import xarray as xr
import glob
import pdb
import os
import datetime as dt


def import_PS_mastertrack_tab(filename):

	"""
	Imports Polarstern master track data during MOSAiC published on PANGAEA. Time
	will be given in seconds since 1970-01-01 00:00:00 UTC and datetime. It also
	returns global attributes in the .tab file so that the information can be
	forwarded to the netcdf version of the master tracks.

	Leg 1, Version 2:
	Rex, Markus (2020): Links to master tracks in different resolutions of POLARSTERN
	cruise PS122/1, Tromsø - Arctic Ocean, 2019-09-20 - 2019-12-13 (Version 2). Alfred
	Wegener Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, 
	PANGAEA, https://doi.org/10.1594/PANGAEA.924668

	Leg 2, Version 2:
	Haas, Christian (2020): Links to master tracks in different resolutions of POLARSTERN
	cruise PS122/2, Arctic Ocean - Arctic Ocean, 2019-12-13 - 2020-02-24 (Version 2).
	Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research,
	Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924674

	Leg 3, Version 2:
	Kanzow, Torsten (2020): Links to master tracks in different resolutions of POLARSTERN
	cruise PS122/3, Arctic Ocean - Longyearbyen, 2020-02-24 - 2020-06-04 (Version 2).
	Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research, 
	Bremerhaven, PANGAEA, https://doi.org/10.1594/PANGAEA.924681

	Leg 4:
	Rex, Markus (2021): Master tracks in different resolutions of POLARSTERN cruise
	PS122/4, Longyearbyen - Arctic Ocean, 2020-06-04 - 2020-08-12. Alfred Wegener 
	Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, PANGAEA,
	https://doi.org/10.1594/PANGAEA.926829

	Leg 5:
	Rex, Markus (2021): Master tracks in different resolutions of POLARSTERN cruise
	PS122/5, Arctic Ocean - Bremerhaven, 2020-08-12 - 2020-10-12. Alfred Wegener
	Institute, Helmholtz Centre for Polar and Marine Research, Bremerhaven, PANGAEA,
	https://doi.org/10.1594/PANGAEA.926910

	Parameters:
	-----------
	filename : str
		Filename + path of the Polarstern Track data (.tab) downloaded from the DOI
		given above.
	"""

	n_prel = 20000		# just a preliminary assumption of the amount of data entries
	reftime = dt.datetime(1970,1,1)
	pstrack_dict = {'time_sec': np.full((n_prel,), np.nan),		# in seconds since 1970-01-01 00:00:00 UTC
					'time': np.full((n_prel,), reftime),		# datetime object
					'Latitude': np.full((n_prel,), np.nan),		# in deg N
					'Longitude': np.full((n_prel,), np.nan),	# in deg E
					'Speed': np.full((n_prel,), np.nan),		# in knots
					'Course': np.full((n_prel,), np.nan)}		# in deg

	f_handler = open(filename, 'r')
	list_of_lines = list()

	# identify header size and save global attributes:
	attribute_info = list()
	for k, line in enumerate(f_handler):
		attribute_info.append(line.strip().split("\t"))	# split by tabs
		if line.strip() == "*/":
			break
	attribute_info = attribute_info[1:-1]	# first and last entry are "*/"

	m = 0		# used as index to save the entries into pstrack_dict
	for k, line in enumerate(f_handler):
		if k > 0:		# skip header
			current_line = line.strip().split()		# split by tabs

			# convert time stamp to seconds since 1970-01-01 00:00:00 UTC:
			pstrack_dict['time_sec'][m] = np.datetime64(current_line[0]).astype('datetime64[s]').astype(np.float64)

			# extract other info:
			pstrack_dict['Latitude'][m] = float(current_line[1])
			pstrack_dict['Longitude'][m] = float(current_line[2])
			pstrack_dict['Speed'][m] = float(current_line[3])
			pstrack_dict['Course'][m] = float(current_line[4])

			m = m + 1

	# truncate redundant lines:
	last_nonnan = np.where(~np.isnan(pstrack_dict['time_sec']))[0][-1] + 1		# + 1 because of python indexing
	for key in pstrack_dict.keys(): pstrack_dict[key] = pstrack_dict[key][:last_nonnan]

	# time to datetime:
	pstrack_dict['time'] = np.asarray([dt.datetime.utcfromtimestamp(tt) for tt in pstrack_dict['time_sec']])

	return pstrack_dict, attribute_info


def save_PS_mastertrack_as_nc(
	export_file,
	pstrack_dict,
	attribute_info):

	"""
	Saves Polarstern master track during MOSAiC to a netCDF4 file.

	Parameters:
	-----------
	export_file : str
		Path where the file is to be saved to and filename.
	pstrack_dict : dict
		Dictionary that contains the Polarstern track information.
	attribute_info : dict
		Dictionary that contains global attributes found in the .tab header.
	"""

	PS_DS = xr.Dataset({'Latitude': 	(['time'], pstrack_dict['Latitude'],
										{'units': "deg N"}),
						'Longitude':	(['time'], pstrack_dict['Longitude'],
										{'units': "deg E"}),
						'Speed':		(['time'], pstrack_dict['Speed'],
										{'description': "Cruise speed",
										'units': "knots"}),
						'Course':		(['time'], pstrack_dict['Course'],
										{'description': "Cruise heading",
										'units': "deg"})},
						coords = 		{'time': (['time'], pstrack_dict['time'],
										{'description': "Time stamp or seconds since 1970-01-01 00:00:00 UTC"})})

	# Set global attributes:
	for attt in attribute_info:
		if (":" in attt[0]) & (len(attt) > 1):
			PS_DS.attrs[attt[0].replace(":","")] = attt[1]
	PS_DS.attrs['Author_of_netCDF'] = "Andreas Walbroel, a.walbroel@uni-koeln.de"

	# encode time:
	encoding = {'time': dict()}
	encoding['time']['dtype'] = 'int64'
	encoding['time']['units'] = 'seconds since 1970-01-01 00:00:00'

	PS_DS.to_netcdf(export_file, mode='w', format="NETCDF4", encoding=encoding)
	PS_DS.close()


"""
	Convert PANGAEA .tab files to netcdf.
"""


path_pstrack = "/example_path/polarstern_track/"


# Polarstern master track data:
pstrack_files = sorted(glob.glob(path_pstrack + "*.tab"))	# should only be one file
for pstrack_file in pstrack_files:
	pstrack_dict, pstrack_att_info = import_PS_mastertrack_tab(pstrack_file)

	ps_export_filename = os.path.join(path_pstrack, os.path.basename(pstrack_file)[:-4] + ".nc")
	save_PS_mastertrack_as_nc(ps_export_filename, pstrack_dict, pstrack_att_info)
