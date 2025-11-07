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
    reftime = np.datetime64("2000-01-01T00:00:00").astype('float32')    # in sec since 1970-01-01 00:00:00 UTC
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
    DS = xr.open_mfdataset(files, concat_dim='time', combine='nested').load()


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
    is_something_here = True        # will check if data is present at index kk of 
                                    # dimension n_hits at any time step
    while is_something_here and (kk < n_hits_max):
        is_something_here = np.any(~np.isnan(DS['lat'].values[:,kk]))
        if is_something_here: kk += 1

    max_n_points = kk

    # truncate the second dimension and removee time indices without data:
    DS = DS.isel(n_hits=np.arange(max_n_points))

    return DS


def read_iasi_step2(path: str, date_range: np.ndarray, RS_DS: xr.Dataset):
    
    """
    Import IASI data for a specific date range that has been processed with manage_iasi.py AND 
    manage_iasi_step2.py. If radiosonde data (RS_DS) is provided, it will be used to compute a  
    height grid for IASI profile data, which is available on pressure levels only. The 
    radiosonde data must have xr.DataArrays of pressurve ("pres") on ("launch_time","height")
    and height ("height") on ("height",) dimensions.
    
    Parameters:
    -----------
    path : str
        Full path of the IASI step2 data.
    date_range : np.ndarray
        Array of np.datetime64 objects covering all days of the date range.
    RS_DS : xr.Dataset
        Radiosonde dataset. Must contain xr.DataArrays of pressurve ("pres") on 
        ("launch_time","height") and height ("height") on ("height",) dimensions.
    """
    
    def first_guess_and_OE_retrieval_flags(DS: xr.Dataset):
        
        """
        Check some more quality flags regarding the first guess (piecewise linear regression) retrieval
        and the physical (optimal estimation) retrieval using the flags flag_fgcheck and flag_retcheck.
        Because several IASI pixels in the vicinity of Polarstern were considered (nhits dimension), 
        the quality flag check must be performed for all pixels. Here, the quality flag of all pixels
        must be good. The flags need to be converted to bit strings to check for the flagged value.

        Parameters:
        -----------
        DS : xarray Dataset
            Dataset containing IASI data (processed with manage_iasi_step2.py) and quality flags.
        """
        
        flag_okay = {'flag_fgcheck': np.full((len(DS.time),), False),
                     'flag_retcheck': np.full((len(DS.time),), False)}
        for flag_value in ['flag_fgcheck', 'flag_retcheck']:
            for k in range(len(DS.time)):
                nonnan_flag = ~np.isnan(DS[flag_value][k,:].values)
                flag_uint = DS[flag_value][k,nonnan_flag].values.astype(np.uint16)
                flag_str = np.asarray([np.binary_repr(val, width=16)[::-1] for val in flag_uint]) # inverting string to have bit 1 at the front
                
                # set flag to okay if water vapour flag is not active:
                flag_okay[flag_value][k] = np.all(np.asarray([flagcheck[1] for flagcheck in flag_str]) == "0")
        
        flag_okay_fg_DA = xr.DataArray(flag_okay['flag_fgcheck'], dims=['time'], coords={'time': (['time'], DS.time.values)})
        flag_okay_ret_DA = xr.DataArray(flag_okay['flag_retcheck'], dims=['time'], coords={'time': (['time'], DS.time.values)})
        
        return flag_okay_fg_DA, flag_okay_ret_DA
    
    
    def get_iasi_quality_flags(DS: xr.Dataset):
        
        flag_okay_fg_DA, flag_okay_ret_DA = first_guess_and_OE_retrieval_flags(DS)
        flag_itconv = ~np.any(DS.flag_itconv < 5, axis=1)
        
        return flag_okay_fg_DA, flag_okay_ret_DA, flag_itconv
    
    
    def apply_flags_and_split_first_guess_and_OE(
        DS: xr.Dataset, 
        var: str,
        reasonable_value_limits: tuple,
        flag_okay_fg_DA: xr.DataArray,
        flag_okay_ret_DA: xr.DataArray, 
        flag_itconv: xr.DataArray):
        
        if var == 'temp':
            var = "atmospheric_temperature_mean"
        elif var == 'q':
            var = "atmospheric_water_vapor_mean"

        data_in_limits_mask = (DS[var] > reasonable_value_limits[0]) & (DS[var] < reasonable_value_limits[1])
        fg_data_in_limits_mask = (DS["fg_"+var] > reasonable_value_limits[0]) & (DS["fg_"+var] < reasonable_value_limits[1])
        iasi_data_okay = xr.where(data_in_limits_mask & flag_okay_ret_DA & flag_itconv,
                                  x=DS[var], y=np.nan)
        iasi_data_fg_okay = xr.where(fg_data_in_limits_mask & flag_okay_fg_DA,
                                     x=DS["fg_"+var], y=np.nan)
        
        return iasi_data_okay, iasi_data_fg_okay
    
    
    def iasi_pressure_to_height_levels(
        iasi_data: xr.DataArray,
        iasi_data_fg: xr.DataArray,
        iasi_pres: np.ndarray,
        sonde_pres: xr.DataArray,
        sonde_height: xr.DataArray,
        ):
        
        sonde_pres_ip = sonde_pres.interp(launch_time=iasi_data.time)
        sonde_pres_ip = sonde_pres_ip.bfill(dim='time')
        sonde_pres_ip = sonde_pres_ip.ffill(dim='time')
        
        iasi_hgt = np.full(iasi_data.shape, np.nan)
        iasi_data_temp = np.full(iasi_data.shape, np.nan)
        for k in range(iasi_data.shape[0]):
            iasi_hgt[k,:] = np.interp(iasi_pres, sonde_pres_ip[k,::-1], sonde_height.values[::-1],
                                      left=np.nan, right=np.nan)
            
            idx_okay = np.where(iasi_pres <= np.nanmax(sonde_pres_ip[k,:]))[0]
            iasi_data_temp[k,idx_okay] = iasi_data.values[k,idx_okay]
            
            if np.all(np.isnan(iasi_data_temp[k,idx_okay])):
                iasi_data_temp[k,idx_okay] = iasi_data_fg[k,idx_okay]
        
        iasi_data[:] = iasi_data_temp
        
        return iasi_hgt, iasi_data
    
    
    files = identify_files_daterange(path, daterange=date_range, file_pattern="PS149_IASI_Polarstern_overlap_*.nc")
    DS = xr.open_mfdataset(files, concat_dim='time', combine='nested').load()
    DS = DS.isel(time=(np.where(
        (~np.any(DS.flag_fgcheck > 511, axis=1)) & 
        (~np.any(DS.fg_qi_atmospheric_water_vapour > 3.95, axis=1))
        )[0]))
    
    
    IASI_pres = {'temp': DS.pressure_levels_temp.values,      # height axis for temperature
                   'q': DS.pressure_levels_humidity.values}     # height axis for q
    reasonable_value_limits = {'temp': (153.15, 333.15),
                               'q': (-0.1,0.1)}
    flag_okay_fg_DA, flag_okay_ret_DA, flag_itconv = get_iasi_quality_flags(DS)
    
    for var in ['temp', 'q']:
        iasi_data, iasi_data_fg = apply_flags_and_split_first_guess_and_OE(DS, var, reasonable_value_limits[var],
                                                                           flag_okay_fg_DA, flag_okay_ret_DA, flag_itconv)
        
        iasi_hgt, iasi_data = iasi_pressure_to_height_levels(iasi_data, iasi_data_fg, IASI_pres[var], 
                                                             RS_DS.pres, RS_DS.height)
        
        DS[var] = iasi_data
        DS['height_'+var] = xr.DataArray(iasi_hgt, dims=['time', 'nl'+var[0]])

    DS = DS.sel(nlt=DS.nlt[::-1], nlq=DS.nlq[::-1])     # surface shall be index 0
    
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
