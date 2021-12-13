#!/usr/bin/env python3
# coding: utf-8
""" intended to be run from the command line, not as a module """

__author__ = "Andrew Davis (addavis@gmail.com)"
__version__ = "0.1 (Jan 2019)"
__license__ = "Distributed under The MIT License (MIT).  See http://opensource.org/licenses/MIT for details."


import os
import nibabel as nib
import numpy as np
import medpy.filter as mpf
import scipy.ndimage as ndi
# import subprocess as sp


def resample_image(data, affine, fwd=True):
    """Resample the image to 1 mm spacing; returns scaled data and adjusted affine.

    Recommend to create a _new_ nifti using the rescaled affine and the orig header."""

    # - handle 4D data

    # Get current voxel scales and calculate resampling factor
    #   uses nibabel dev strategy (https://github.com/nipy/nibabel/issues/670)
    RZS = affine[:3, :3]                            # rotation, zoom, skew portion
    old_scales = np.sqrt(np.sum(RZS**2, axis=0))    # should give the same values as nib.affines.voxel_sizes(hdr.get_best_affine()) and hdr.get_zooms(); gets scale factors from magnitudes of col vectors in affine (see https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati)
    target_scales = np.array([1.]*3)

    if fwd:
        img_scale_fac = old_scales/target_scales

    else:
        img_scale_fac = target_scales/old_scales


    # Rescale data using the zoom factor
    new_data = ndi.zoom(data, img_scale_fac, order=3, mode='reflect')


    # Rescale affine to match
    aff_scale_fac = 1./img_scale_fac

    new_aff = np.array(affine, copy=True)

    if fwd:
        new_aff[:3, :3] = RZS*np.diag(aff_scale_fac)

    else:
        new_aff[:3, :3] = RZS*np.diag(img_scale_fac)*np.diag(aff_scale_fac)


    # Take care of the translation
    #   this is necessary since the (0,0,0) co-ord in voxel space is the _center_ of the first voxel (see http://nipy.org/nibabel/coordinate_systems.html)
    T = affine[:3, 3]   # x,y,z translations as vector
    T_sign = np.sign(np.diagonal(affine[:3, :3]))

    if fwd:
        # negative values for going to higher resolution
        T_offset = 0.5*old_scales*(aff_scale_fac - 1)   # x,y,z correction based on my diagram
        new_aff[:3, 3] = T + T_offset*T_sign

    else:
        T_offset = 0.5*old_scales*(img_scale_fac - 1)
        new_aff[:3, 3] = T - T_offset*T_sign + T_offset*T_sign  # pointless, right? Yet...


    return new_data, new_aff

    # **old idea** -- less reliable way from medpy didn't properly handle sform/qform
    # data_res, hdr_res = mpf.resample(data, hdr, target_scales)
    # the resampled images from mpf don't work well in fsleyes -- it appears mpf is not changing
    #   the header to account for the zooms

    # **old idea 2** -- more hassle and hacky way that cleans up set_zooms' deficiencies
    # header.set_zooms([x,y,z]), sets the pixdim; can just use this with ndi.zoom?
    # header.set_qform(affine, code), may be necessary otherwise
    # hdr.set_zooms(target_scales)  # changes qform only, regardless of codes!
                                    # see http://nipy.org/nibabel/nifti_images.html for codes
    # hdr.set_sform(hdr.get_qform(), code=1)


def gauss_smooth(data):
    """smooth the image isotropically with sigma ~ 2.1; returns smoothed data"""

    # note FWHM = 2.355 sigma
    # FWHM of twice the voxel dimensions is often considered a reasonable compromise in fMRI
    # So, now that the data has been resampled to 1 mm...

    sigma = 1.0*2/2.355

    new_data = ndi.gaussian_filter(data, sigma, mode='reflect')

    return new_data


def anis_smooth(data, affine):
    """run the anisotropic diffusion filter from medpy; returns smoothed data"""

    # medpy function expects float valued data (0 -> 1)
    a = np.min(data)
    b = np.max(data)
    flv_data = (data - a)/(b - a)

    # anisotropic_diffusion(data, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1)
    new_data = mpf.anisotropic_diffusion(flv_data, niter=3,
                                         voxelspacing=nib.affines.voxel_sizes(affine))

    # get voxel values back
    new_data = new_data*(b - a) + a

    return new_data


def make_nii(new_data, new_aff=None, new_hdr=None):
    """Create a new nii object; use orig affine, units by default; returns nii"""

    if new_hdr is None:
        new_nii = nib.Nifti1Image(new_data, new_aff)
        new_nii.header.set_xyzt_units('mm', 'sec')      # or nii.get_xyzt_units()
        new_nii.set_data_dtype(np.float32)
        new_nii.set_qform(None, code=1)
        new_nii.set_sform(None, code=1)

    else:
        # if new_aff is None, affine will come from the header
        new_nii = nib.Nifti1Image(new_data, new_aff, header=new_hdr)
        new_nii.set_sform(None, code=1)     # default was 2, and qform zeroed out and code=0

    new_nii.header.structarr['descrip'] = b'CAN-BIND DMRI'

    return new_nii


def main(nii_fn, out_file):
    """Reads a nii image (3D or 4D).

    - resamples to 1 mm resolution
    - smoothes with gaussian and anisotropic (perona-malik) diffusion
    - returns to input resolution

    Outputs a new nii volume after processing.

    Example:
    """

    # Check args
    assert os.path.isfile(nii_fn), "File not found: {}".format(nii_fn)

    if out_file is None:
        out_file = nii_fn.replace('.nii', '-ablur.nii')
    # assert not os.path.exists(out_file), "file exists: {}".format(out_file)


    # Generate 4D float data array with time axis being DW volumes
    # nii_fn = '../t2w_epi_mean.nii.gz'
    nii = nib.load(nii_fn)
    data = nii.get_fdata()                  # data.shape -> (96, 96, 52, 37)

    assert data.dtype == np.float_, "Expected float data from nifti."
    assert len(data.shape) <= 4, "Expected 4D data at the most."


    # Resample image
    data_res, aff_res = resample_image(data, nii.affine)
    make_nii(data_res, new_aff=aff_res).to_filename('res.nii.gz')


    # Smooth data
    if len(data.shape) == 4:
        # handle volumes individually
        for v in range(data.shape[3]):
            smdata = anis_smooth(data_res[:,:,:,v], aff_res)

            make_nii(smdata, new_aff=nii.affine).to_filename(out_file.replace('.nii', v + '.nii'))

    elif len(data.shape) == 3:
        gdata = gauss_smooth(data_res)
        make_nii(gdata, new_aff=aff_res).to_filename('gdata.nii.gz')

        adata = anis_smooth(data_res, aff_res)
        make_nii(adata, new_aff=aff_res).to_filename('adata.nii.gz')

        adata2 = anis_smooth(gdata, aff_res)
        make_nii(adata2, new_aff=aff_res).to_filename('adata2.nii.gz')

        # make_nii(smdata, new_aff=nii.affine).to_filename(out_file)


    # inverse resample operation
    data2, aff2 = resample_image(adata2, nii.affine, fwd=False)
    make_nii(data2, new_aff=aff2).to_filename('adata_invres.nii.gz')



# Entry point
if __name__ == '__main__':

    # Parse args and run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nii_fn", metavar='nii', help="nifti file to blur")
    parser.add_argument("-o", "--outfile", dest='out_file', metavar="OF", help="output file path of blurred image")
    args = parser.parse_args()

    # nii.gz output implied
    if 'nii' not in args.out_file:
        args.out_file = args.out_file + '.nii.gz'

    main(args.nii_fn, args.out_file)

