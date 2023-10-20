import numpy as np
import datetime as dt
import pandas as pd
import xarray as xr
import pdb
import os
import glob
import sys


def import_iasi_nc(
	files):

	"""
	Import IASI file(s) and return as xarray dataset. Also cuts irrelevant variables and
	dimensions. Also the time needs to be decoded manually.

	Parameters:
	-----------
	files : list of str
		Path + filenames of the netCDF files to be imported. 
	"""

	def cut_vars(DS):

		unwanted_vars = ['forli_layer_heights_co', 'forli_layer_heights_hno3', 'forli_layer_heights_o3', 'brescia_altitudes_so2', 'solar_zenith', 
					'satellite_zenith', 'solar_azimuth', 'satellite_azimuth', 'fg_atmospheric_ozone', 'fg_qi_atmospheric_ozone', 'atmospheric_ozone', 
					'integrated_ozone', 'integrated_n2o', 'integrated_co', 'integrated_ch4', 'integrated_co2', 'surface_emissivity', 
					'number_cloud_formations', 'fractional_cloud_cover', 'cloud_top_temperature', 'cloud_top_pressure', 'cloud_phase', 
					'instrument_mode', 'spacecraft_altitude', 'flag_cdlfrm', 'flag_cdltst', 'flag_daynit', 'flag_dustcld', 
					'flag_numit', 'flag_nwpbad', 'flag_physcheck', 'flag_satman', 'flag_sunglnt', 'flag_thicir', 'nerr_values', 'error_data_index', 
					'temperature_error', 'water_vapour_error', 'ozone_error', 'co_qflag', 'co_bdiv', 'co_npca', 'co_nfitlayers', 'co_nbr_values', 
					'co_cp_air', 'co_cp_co_a', 'co_x_co', 'co_h_eigenvalues', 'co_h_eigenvectors', 'hno3_qflag', 'hno3_bdiv', 'hno3_npca', 
					'hno3_nfitlayers', 'hno3_nbr_values', 'hno3_cp_air', 'hno3_cp_hno3_a', 'hno3_x_hno3', 'hno3_h_eigenvalues', 'hno3_h_eigenvectors',
					'o3_qflag', 'o3_bdiv', 'o3_npca', 'o3_nfitlayers', 'o3_nbr_values', 'o3_cp_air', 'o3_cp_o3_a', 'o3_x_o3', 'o3_h_eigenvalues', 
					'o3_h_eigenvectors', 'so2_qflag', 'so2_col_at_altitudes', 'so2_altitudes', 'so2_col', 'so2_bt_difference', 'fg_surface_temperature',
					'fg_qi_surface_temperature', 'surface_temperature']
		remaining_unwanted_dims = ['cloud_formations', 'nlo', 'new']

		# check if unwanted_vars exist in the dataset (it may happen that not all variables 
		# exist in all IASI files):
		uv_exist = np.full((len(unwanted_vars),), False)
		for i_u, u_v in enumerate(unwanted_vars):
			if u_v in DS.variables:
				uv_exist[i_u] = True

		DS = DS.drop_vars(np.asarray(unwanted_vars)[uv_exist])
		DS = DS.drop_dims(remaining_unwanted_dims)
		return DS

	DS = xr.open_mfdataset(files, combine='nested', concat_dim='along_track', preprocess=cut_vars, decode_times=False)

	# decode time:
	reftime = np.datetime64("2000-01-01T00:00:00").astype('float32')	# in sec since 1970-01-01 00:00:00 UTC
	record_start_time_npdt = (DS.record_start_time.values + reftime).astype('datetime64[s]')
	record_stop_time_npdt = (DS.record_stop_time.values + reftime).astype('datetime64[s]')
	DS['record_start_time'] = xr.DataArray(record_start_time_npdt, dims=['along_track'], attrs={'long_name': "Record start time", 
											'units': "seconds since 1970-01-01 00:00:00 UTC"})
	DS['record_stop_time'] = xr.DataArray(record_stop_time_npdt, dims=['along_track'], attrs={'long_name': "Record stop time", 
											'units': "seconds since 1970-01-01 00:00:00 UTC"})

	# also rename integrated water vapour variable:
	DS = DS.rename({'integrated_water_vapor': 'iwv'})

	return DS


def import_iasi_step1(
	path_data):

	"""
	Imports IASI step 1 data processed with manage_iasi.py that lie within the path given. The data will be 
	returned as xarray dataset sorted by time. The second dimension 'n_points' will be truncated as far as possible. 
	The initial size of this dimension was only a proxy to cover as many identified IASI pixels for one 
	Polarstern track time step as possible.

	Parameters:
	-----------
	path_data : str
		Data path as string indicating the location of the processed IASI files. No subfolders will be searched.
	"""

	# identify files:
	files = sorted(glob.glob(path_data + "*.nc"))
	DS = xr.open_mfdataset(files, concat_dim='time', combine='nested')


	# adjustment of time; and constrain to the specified date range:
	DS = DS.sortby(DS.time)

	# check if time duplicates exist:
	if np.any(np.diff(DS.time.values) == np.timedelta64(0, "ns")):
		raise RuntimeError("It seems that the processed IASI data has some duplicates. Removing them has not yet been coded. Have fun.")
	

	# check how much of the dimension 'n_hits' is really needed: sufficient to check only one 
	# of the variables with this second dimension because all others use this dimension similarly:
	max_n_points = -1
	n_hits_max = len(DS.n_hits)

	# loop through all columns (n_hits dimension) and check if data still exists:
	kk = 0
	is_something_here = True		# will check if data is present at index kk of 
									# dimension n_hits at any time step
	while is_something_here and (kk < n_hits_max):
		is_something_here = np.any(~np.isnan(DS['lat'].values[:,kk]))
		if is_something_here: kk += 1

	max_n_points = kk

	# truncate the second dimension and removee time indices without data:
	DS = DS.isel(n_hits=np.arange(max_n_points))

	return DS


def import_PS_mastertrack(filename):

	"""
	Imports Polarstern master track data during MOSAiC published on PANGAEA. Time
	will be given in seconds since 1970-01-01 00:00:00 UTC and datetime. It also
	returns global attributes in the .tab file so that the information can be
	forwarded to the netcdf version of the master tracks.

	Parameters:
	-----------
	filename : list of str
		List containing path + filenames of the Polarstern Track data (.nc).
	"""

	if type(filename) == list:
		DS = xr.open_mfdataset(filename, combine='nested', concat_dim='time')
		return DS
