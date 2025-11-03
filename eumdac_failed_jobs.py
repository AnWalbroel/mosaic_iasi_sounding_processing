import os
import sys
import shutil
import glob
import pdb
import subprocess
import datetime as dt

wdir = os.getcwd() + "/"

import numpy as np


def eumdac_download(
    list_files,
    path_out,
    path_yaml):

    """
    Downloading EUMETSAT data products using eumdac download (here, specified for a IASI 
    data collection: IASI Combined Sounding Products - Metop) to obtain files where the download
    failed in the first attempt. The files will be saved to path_out

    Parameters:
    -----------
    list_files : list of str
        List of strings containing the EUMETSAT data files (or products) which are to be 
        downloaded and processed according to --tailor and the yaml file.
    path_out : str
        Path where the downloaded data is to be saved to.
    path_yaml : str
        Path where to find the iasi_mosaic.yaml file, which indicates the processing of the 
        downloaded file.
    """

    n_files = len(list_files)

    for fi in list_files:
        print(f"Running eumdac download -c EO:EUM:DAT:METOP:IASSND02 -p {fi} --tailor {path_yaml}iasi_mosaic.yaml -o {path_out}")

        output = subprocess.run(["eumdac", "download", "-c", "EO:EUM:DAT:METOP:IASSND02", "-p", fi,
                                "--bbox", "-180.0", "70.0", "180.0", "90.0",
                                "--tailor", f"{path_yaml}iasi_mosaic.yaml", "-o", path_out], 
                                capture_output=True, timeout=1800)  # timeout in s: max time for attempting download

        if output.stdout.decode("ascii").find("has been downloaded") != -1:
            print("Download was successful....")
            n_files -= 1

        else:
            print("File either already existed and hasn't been documented correctly in the log file or the download failed....")

            # A file may already exist when, for example, another eumdac download has been requested for subsequent or preceding
            # days.

    return n_files


"""
    This script intends to identify failed jobs when downloading IASI data with EUMETSAT's API
    'eumdac download -c EO:EUM:DAT:METOP:IASSND02 ...'. The job numbers where 'has been downloaded'
    has never been displayed failed and must be identified. 
    - open log file
    - cycle through log file to identify job numbers
    - inquire if job has been downloaded successfully
    - identify filenames of failed jobs
    (- subprocess to download the missing files?)
"""


# paths:
path_logs = "/mnt/e/VAMPIRE2/IASI/logs/"
path_logs_done = os.path.join(path_logs, "done")
path_iasi_yaml = "/mnt/e/VAMPIRE2/IASI/"
path_iasi_out = "/mnt/e/VAMPIRE2/IASI/raw/"


# settings:
set_dict = {
            'download_failed': True,        # if True, failed jobs are attempted to be downloaded
            'identify_unidentified': True,  # if True, unidentified failed jobs are attempted to be found and downloaded
            }


# identify log files:
log_files = sorted(glob.glob(path_logs + f"eumdac_*.txt"))
os.makedirs(path_logs_done, exist_ok=True)

for log_file in log_files:

    print(f"Processing LOG FILE {log_file}....")

    # get query_start and query_end times from the filename:
    filename = os.path.basename(log_file)
    filename_split = filename.split("-")
    set_dict['query_start'] = dt.datetime.strptime(filename_split[0][-13:],
                                                    "%Y%m%dT%H%M").strftime("%Y-%m-%dT%H:%M") + ":00"
    set_dict['query_end'] = dt.datetime.strptime(filename_split[1][:13],
                                                    "%Y%m%dT%H%M").strftime("%Y-%m-%dT%H:%M") + ":59"


    # open log file:
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f_handler:

        n_jobs_detected = False     # checks if the number of jobs has already been detected
        for k, line in enumerate(f_handler):

            # identify number of jobs:
            if not n_jobs_detected:
                line_split = line.split(" ")
                try:
                    if "Processing" in line_split:
                        set_dict['n_jobs'] = int(line_split[1])
                        print(f"Number of jobs found: {set_dict['n_jobs']}")
                        n_jobs_detected = True

                        # initialize arrays to save job numbers and completeness status:
                        job_no = (np.ones((set_dict['n_jobs'],))*(-1)).astype(np.int32)
                        completed = np.full((set_dict['n_jobs'],), False)    # if True, data has been suceessfully downloaded

                        filename_dummy = "IASI_SND_02_M00_00000000000000Z_00000000000000Z_N_O_00000000000000Z"
                        n_dummy = len(filename_dummy)
                        filenames = np.full((set_dict['n_jobs'],), filename_dummy)

                except:
                    print(f"Couldn't find the number of jobs in {filename}. Happy debugging!")
                    pdb.set_trace()
                

            # identify job number:
            id_str_0 = "Job "       # strings to identify job number
            id_str_1 = ":"
            jn_0 = line.find(id_str_0)
            jn_1 = line.find(id_str_1)
            n_id_str_0 = len(id_str_0)

            if (jn_0 > -1) and (jn_1 > -1): # then, the correct line type has been found:
                job_ = int(line[jn_0+n_id_str_0:jn_1])
                job_no[job_-1] = job_

                # check if that job has been downloaded:
                id_str_2 = "has been downloaded"
                id_str_file_exists = "File exists"
                jn_2 = line.find(id_str_2)
                jn_file_exists = line.find(id_str_file_exists)
                if (jn_2 > -1) | (jn_file_exists > -1):
                    completed[job_-1] = True

                # identify filenames:
                id_str_3 = "IASI_SND_02_"
                n_id_str_3 = len(id_str_3)
                jn_3 = line.find(id_str_3)
                if jn_3 > -1:
                    filenames[job_-1] = line[jn_3:jn_3+n_dummy]

        # identify failed jobs:
        idx_failed = np.where((~completed) & (job_no > -1))[0]
        job_no_failed = job_no[idx_failed]
        filenames_failed = filenames[idx_failed]

        # also jobs can remain unidentified (no filenames, ...):
        unident_jobs = np.where(job_no == -1)[0]
        if len(unident_jobs) > 0:
            unident_jobs += 1

    # create output path if needed (to save downloaded IASI files):
    out_dir = os.path.dirname(path_iasi_out)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # attempt to get the missing files:
    if set_dict['download_failed']:
        n_failed = len(filenames_failed)
        print(f"Identified {n_failed} failed downloads. Attempting to download missing files....")
        n_failed = eumdac_download(filenames_failed, path_iasi_out, path_iasi_yaml)

        # conclude:
        if n_failed == 0:
            print("No missing files or missing files have been successfully downloaded.")

            print("Remaining unidentified incompleted job numbers: ")
            for u_job in unident_jobs: print(u_job)


    if set_dict['identify_unidentified']:
        # identify files for the collection and identify files not listed in "filenames":
        print("Searching for unidentified files....")

        # find identified jobs before and after the unidentified one:
        start_time = set_dict['query_start']
        end_time = set_dict['query_end']


        # search for potential files:
        print(f"Running eumdac search -c EO:EUM:DAT:METOP:IASSND02 -s {start_time} -e {end_time}")
        output = subprocess.run(["eumdac", "search", "-c", "EO:EUM:DAT:METOP:IASSND02", "-s", start_time,
                                "-e", end_time], capture_output=True, timeout=1800)
        query_files = np.asarray(output.stdout.decode("ascii").split("\n"))

        # loop through query files and check if that file exists in filenames:
        file_existing = np.full((len(query_files),), False)
        for i_q, query_file in enumerate(query_files):
            if query_file:  # empty elements in query files may exist
                if query_file in filenames:
                    file_existing[i_q] = True
            else:
                file_existing[i_q] = True


        # identify unidentified files and try to retrieve them:
        unident_files = query_files[~file_existing]
        n_unident = len(unident_files)
        if n_unident > 0: 
            print("Attempting to download unidentified files....")
            n_unident = eumdac_download(unident_files, path_iasi_out, path_iasi_yaml)


        # conclude:
        if n_unident == 0:
            print("No unidentified files or unidentified files have been successfully downloaded.")

    shutil.move(log_file, os.path.join(path_logs_done, os.path.basename(log_file)))
