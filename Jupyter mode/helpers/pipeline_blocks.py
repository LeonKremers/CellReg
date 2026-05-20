# helper functions to run the pipeline for multiple scans and to grab the CellReg output
# designed to run with the adamacs datajoint pipeline
# Author: Leone Kremers, 2026

from adamacs.pipeline import (
    session,
    event,
    imaging,
    scan,
)

import numpy as np
import os
import scipy.io
import mat73


def define_scans(samesite_id, stim_type, paramset, behavior_stage=None):
    """
    Function to define scans for a given same_site_id and stimulus type.

    Args:
        same_site_id (str): same_site_id
        stim_type (str): stimulus type (e.g. "GRATING")
        paramset (int): suite2P paramset index
        behavior_stage (str): optional string to limit defined scans to a certain datajoint behavior stage

    Returns:
        scans (list): list of scans
        curation_keys (dict): dictionary containing all curation keys for all scans
        samesite_scan_key (list): list of scan keys
    """

    samesite_session_key = (
        session.Session * session.SessionSameSite & f'same_site_id = "{samesite_id}"'
    ).fetch("KEY")
    # Define scans and select onyl scans where the scan_notes = stimulus type
    bad_string = "Do not use!"
    # limit to scans with behavior stage string
    if behavior_stage is not None:
        samesite_scan_key = (
            scan.Scan * event.BehaviorRecording
            & samesite_session_key
            & f'scan_notes LIKE "%{stim_type}%"'
            & f'scan_notes NOT LIKE "%{bad_string}%"'
            & f'recording_notes LIKE "%{behavior_stage}%"'
        ).fetch("KEY")
    # get all scans
    else:
        samesite_scan_key = (
            scan.Scan
            & samesite_session_key
            & f'scan_notes LIKE "%{stim_type}%"'
            & f'scan_notes NOT LIKE "%{bad_string}%"'
        ).fetch("KEY")
    scans = (scan.Scan & samesite_scan_key).fetch("scan_id")

    # Get curation keys for latest curations
    curation_keys = {}
    for i in range(len(samesite_scan_key)):
        scan_key = samesite_scan_key[i]
        latest_curation = (imaging.Curation & scan_key).fetch("curation_id").max()
        curation_keys[i] = (
            imaging.Curation
            & scan_key
            & f"curation_id={latest_curation}"
            & f"paramset_idx={paramset}"
        ).fetch("KEY")[0]

    print("Scans:")
    print(scans)

    return scans, curation_keys, samesite_scan_key


def import_CellReg_output(samesite_id, stim_type):
    """
    Function to import CellReg output from file.

    Args:
        same_site_id (str): same_site_id
        stim_type (str): stimulus type (e.g. "GRATING")

    Returns:
        cell_to_index_map (array): array containing all mask IDs for each session. Each row is one roi_group and each column is one session. If a mask is present in a session the corresponding entry is the mask ID, if not the entry is 0.
                                    (Beware of Matlab numbering! Perform -1 to get Python numbering)
    """

    # Import CellReg output
    mouse_id = (session.Session & f'session_id = "{samesite_id}"').fetch("subject")[0]
    results_path = (
        "/datajoint-data/data/leonk/analysis/cellreg/LE_"
        + mouse_id
        + "_"
        + samesite_id
        + "_"
        + stim_type
        + "_CR/Results/"
    )
    name_includes = "cellRegistered"

    temp = False
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if name_includes in file:
                if temp == True:
                    print("Multiple CellReg outputs found: Importing latest one")
                file_name = results_path + file
                temp = True
    if temp == True:
        print("Now importing cell_registered.mat")
        try:
            # import .mat file if version <7.3
            cellRegistered = scipy.io.loadmat(file_name)
        except NotImplementedError:
            # import .mat file if version = 7.3
            cellRegistered = mat73.loadmat(file_name)
        except:
            raise ValueError("Could not read .mat file at all...")

        # Convert CellReg output to array
        cell_to_index_map = cellRegistered["cell_registered_struct"][
            "cell_to_index_map"
        ]

    if temp == False:
        raise ValueError("No CellReg output found")

    return cell_to_index_map


def get_indices(samesite_id, scans, curation_keys, stim_type):
    """
    Function to grab CellReg output from Datajoint.
    Grabs all possible indices which are presdent across all sessions and additionally all indices are present across all sessions and classified as soma in all sessions.

    Args:
        same_site_id (str): same_site_id
        scans (list): list of scans
        curation_keys (dict): dictionary containing all curation keys for all scans
        stim_type (str): stimulus type (e.g. "GRATING")

    Returns:
        indices (list): list of all roi_groups which have masks present across all sessions
        neuron_only_indices (list): list of all roi_groups which have masks present across all sessions and which are classified as soma in all sessions

    """

    bad_string = "Do not use!"
    # fetch all groups present in the same_site session with defined stimulus
    samesite_session_key = (
        session.Session * session.SessionSameSite & f'same_site_id = "{samesite_id}"'
    ).fetch("KEY")
    groups = np.unique(
        (
            scan.Scan * imaging.Segmentation.Mask
            & samesite_session_key
            & f'scan_notes LIKE "%{stim_type}%"'
            & f'scan_notes NOT LIKE "%{bad_string}%"'
        ).fetch("roi_group")
    )

    indices = []
    mask_ids = []
    neuron_only_indices = []
    neuron_only_mask_ids = []

    # loop through all groups and fetch all masks that have the same roi_group
    for i in range(len(groups)):
        masks = []
        mask_types = []
        for j in range(len(scans)):
            curation_key = curation_keys[j]
            mask = (
                imaging.Segmentation.Mask & curation_key & f'roi_group = "{i}"'
            ).fetch("mask")
            if mask.shape[0] > 0:
                masks.append(mask[0])
                mask_types.append(
                    (
                        imaging.Segmentation.Mask * imaging.MaskClassification.MaskType
                        & curation_keys[j]
                        & f"mask={mask[0]}"
                    ).fetch("mask_type")
                )
        # only append indices that are present across all sessions
        if len(masks) == len(scans):
            indices.append(i)
            mask_ids.append(masks)
            # only append indices which are classified as soma in all session
            if all(mask_type.size > 0 for mask_type in mask_types):
                if mask_types == ["soma"] * len(scans):
                    neuron_only_indices.append(i)
                    neuron_only_mask_ids.append(masks)
    mask_ids = np.array(mask_ids)
    neuron_only_mask_ids = np.array(neuron_only_mask_ids)

    # print("Possible indices are:")
    # print(np.array(indices))
    # print(len(indices))
    # print("")
    # print("Possible neuron only indices are:")
    # print(np.array(neuron_only_indices))
    # print(len(neuron_only_indices))

    return indices, neuron_only_indices


def get_all_mask_ids(samesite_id, scans, curation_keys, stim_type):
    """
    Function to get all mask IDs from each session.
    Args:
        same_site_id (str): same_site_id
        scans (list): list of scans
        curation_keys (dict): dictionary containing all curation keys for all scans
        stim_type (str): stimulus type (e.g. "Gratings")
    Returns:
        mask_ids_all (array): array containing all mask IDs for each session, if no mask is present for the roi_group in a certain session the ID is set to -1
    """

    bad_string = "Do not use!"
    # fetch all groups present in the same_site session with defined stimulus
    samesite_session_key = (
        session.Session * session.SessionSameSite & f'same_site_id = "{samesite_id}"'
    ).fetch("KEY")
    groups = np.unique(
        (
            scan.Scan * imaging.Segmentation.Mask
            & samesite_session_key
            & f'scan_notes LIKE "%{stim_type}%"'
            & f'scan_notes NOT LIKE "%{bad_string}%"'
        ).fetch("roi_group")
    )

    mask_ids_all = np.full((len(groups), len(scans)), -1)
    for i in range(len(scans)):
        for j in range(len(groups)):
            try:
                mask_ids_all[j, i] = (
                    imaging.Segmentation.Mask & curation_keys[i] & f'roi_group = "{j}"'
                ).fetch("mask")
            except:
                mask_ids_all[j, i] = -1

    return mask_ids_all


def get_all_neuron_ids(samesite_id, scans, curation_keys, stim_type):
    """
    Function to get all neuron IDs from each session.

    Args:
        same_site_id (str): same_site_id
        scans (list): list of scans
        curation_keys (dict): dictionary containing all curation keys for all scans
        stim_type (str): stimulus type (e.g. "GRATING")

    Returns:
        neuron_ids_all (array): array containing all neuron IDs for each session, if no neuron is present for the roi_group in a certain session the ID is set to -1

    """

    bad_string = "Do not use!"
    # fetch all groups present in the same_site session with defined stimulus
    samesite_session_key = (
        session.Session * session.SessionSameSite & f'same_site_id = "{samesite_id}"'
    ).fetch("KEY")
    groups = np.unique(
        (
            scan.Scan * imaging.Segmentation.Mask
            & samesite_session_key
            & f'scan_notes LIKE "%{stim_type}%"'
            & f'scan_notes NOT LIKE "%{bad_string}%"'
        ).fetch("roi_group")
    )

    neuron_ids_all = np.full((len(groups), len(scans)), -1)

    for i in range(len(scans)):
        mask_LUT = (
            imaging.Segmentation.Mask * imaging.MaskClassification.MaskType
            & curation_keys[i]
        ).fetch("mask", "mask_type")
        for j in range(len(groups)):
            # convert mask ID to neuron count / id
            try:
                # if neuron part of the roi_group in this session
                neuron_ids_all[j, i] = np.where(
                    mask_LUT[0]
                    == (
                        imaging.Segmentation.Mask
                        & curation_keys[i]
                        & f'roi_group = "{j}"'
                    ).fetch("mask")
                )[0][0]
            except:
                # if neuron not part of the roi_group in this session
                neuron_ids_all[j, i] = -1

    return neuron_ids_all
