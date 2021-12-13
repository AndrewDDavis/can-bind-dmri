#!/usr/bin/env python3
# coding: utf-8
""" intended to be run from the command line, not as a module """

__author__ = "Andrew Davis (addavis@gmail.com)"
__version__ = "0.1 (Jan 2019)"
__license__ = "Distributed under The MIT License (MIT).  See http://opensource.org/licenses/MIT for details."


import plac
# import os.path
import nibabel as nib
import numpy as np
# import subprocess as sp


@plac.annotations(
 dwi_nii="nifti file",
 bval_file="bvals file (FSL style)",
 out_file="output file")

def main(dwi_nii, bval_file, out_file=None):
    """Reads a concatenated DWI nifti 4D volume and a matching bval file in FSL style (i.e. row).

    Output a nii volume with each voxel being the signal ratio of b=0 to b=1000 images. 

    Example: dmri_sigts_stats.py 'dwi_merged-ecc.nii.gz' '../dti.bval'
    """

    # Check args
    assert os.path.isfile(dwi_nii), "File not found: {}".format(dwi_nii)
    assert os.path.isfile(bval_file), "File not found: {}".format(bval_file)
    # if out_file is None:
    #     out_file = "selected_frames.nii"
    # assert not os.path.exists(out_file), "file exists: {}".format(out_file)

    # Grab dwi_nii TR
    # TR = sp.check_output(["fslval", dwi_nii, "pixdim4"]).strip()
    #print "TR={}".format(TR)


    # Generate 4D data array with time axis being DW volumes
    # dwi_nii='dwi_merged-ecc.nii.gz'
    nii = nib.load(dwi_nii)
    data = nii.get_fdata()                  # data.shape -> (96, 96, 52, 37)

    # Generate boolean arrays for b=0 volumes and DW volumes
    # bval_file = '../dti.bval'
    bvals = np.genfromtxt(bval_file)    # 1D vector
    b0_vols = bvals == 0                # boolean

    # For each voxel, get the mean signal ratio across b=0 and DW volumes, put it in a 4D array
    sigrat_data = np.zeros(data.shape[0:3], dtype=data.dtype)
    sigsub_data = np.zeros(data.shape[0:3], dtype=data.dtype)

    if b0_vols.sum() > 1:
        sigfancy = True
        sigfancy_data = np.zeros(data.shape[0:3], dtype=data.dtype)
    else:
        print("No sigfancy will be output, since only 1 b=0 volume")

    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            print("i,j,k: {},{},{}".format(i,j,k))
            # take the ratio of signals according to the boolean
            d0 = data[i, j, k, b0_vols]
            d1 = data[i, j, k, ~b0_vols]

            sigrat_data[i, j, k] = d0.mean()/d1.mean()

            sigsub_data[i, j, k] = (d0.mean() - d1.mean())/d1.std()

            # check if we have the b=0 std as well
            if sigfancy:
                sigfancy_data[i, j, k] = (d0.mean() - d1.mean())/d0.std()


    # Write out new niftis
    def make_nii(imgdata):
        """Create a new nii object with the same affine, units as input"""
        out_nii = nib.Nifti1Image(imgdata, nii.affine)
        out_nii.header.set_xyzt_units('mm', 'sec')
        out_nii.set_data_dtype(np.float32)
        out_nii.set_qform(None, code=1)
        out_nii.set_sform(None, code=1)

    make_nii(sigrat_data).to_filename('sigrat.nii.gz')
    make_nii(sigsub_data).to_filename('sigsub.nii.gz')
    make_nii(sigfancy_data).to_filename('sigfancy.nii.gz')


    # sigrat:
    # - most brain values above 1.375
    # - most non-brain below 2

    # sigsub:
    # - most brain values above 1.5
    # - most non-brain below 2

    # sigfancy:
    # - most brain values above 2
    # - most non-brain below 2

    # consider gaussian blurs on the input data or sigrat etc...


# Entry point
if __name__ == '__main__':
    plac.call(main)

