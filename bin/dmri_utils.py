#!/usr/bin/env python3
# coding: utf-8
"""Various utility functions and config variables for working with diffusion MRI on the CAN-BIND project.

Gets imported from dmri_preprocess et al as: import dmri_utils as du
"""

__author__ = "Andrew Davis (addavis@gmail.com)"
__version__ = "0.2 (Jan 2021)"
__license__ = "Distributed under The MIT License (MIT).  See http://opensource.org/licenses/MIT for details."

import os, tempfile, shutil
from os.path import dirname, abspath
from pathlib import Path
import subprocess as sp
from glob import glob
import ipdb

import numpy as np
import nibabel as nib
from scipy import stats
import scipy.ndimage as ndi
from skimage.morphology import ball
from skimage.filters import threshold_minimum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Config vars set and accessed from other scripts as du.foo
nii = None           # the input nifti object
verbose = None       # controls verbosity of output to terminal
bvals = None         # 1D vector corresponding to nii volumes
t2w_vols = None      # boolean vector
dw_vols = None
nzbvals_uniq = None  # array of non-zero bvals, e.g. [1000, 3000]
dw1_vols = None      # for the first shell, regardless of the number of shells

# Temp directory for intermediate files
tmpd = tempfile.mkdtemp()

# Modify environment to output plain niis in tmpd for speed
nogz_env = os.environ.copy()

assert 'FSLDIR' in nogz_env, "FSLDIR not set in envirionment"

if '_GZ' in nogz_env['FSLOUTPUTTYPE']:
    output_gzip = True
else:
    output_gzip = False

nogz_env['FSLOUTPUTTYPE'] = 'NIFTI'

# local script dir (normally shell environment var)
if not ('DMRIDIR' in nogz_env):
    dmridir = dirname(abspath(__file__))
    nogz_env['DMRIDIR'] = dmridir
    print(f"dmri_utils: DMRIDIR set to {dmridir}")

if not ('MRTRIXDIR' in nogz_env):
    mrtrixdir = shutil.which('dwidenoise').rsplit('/bin')[0]
    nogz_env['MRTRIXDIR'] = mrtrixdir
    print(f"dmri_utils: MRTRIXDIR set to {mrtrixdir}")


def check_load_nii(nii_fn):
    """Check nii_fn to make sure it exists, adding .nii and .nii.gz to the path as necessary. Then
    load the file using nibabel and return the NIfTI object.
    """

    nii_fnp = get_nii_prefix(nii_fn)

    if Path(nii_fnp + '.nii').is_file() and Path(nii_fnp + '.nii.gz').is_file():
        raise RuntimeError(f"Found both .nii and .nii.gz with prefix {nii_fnp}")

    # need filename to be a string, not Path object for tricks below
    if not isinstance(nii_fn, str):
        nii_fn = str(nii_fn)

    nii_fn_orig = nii_fn

    while True:
        try:
            newnii = nib.load(nii_fn)

        except FileNotFoundError:
            # nii specified without extension?
            if not ('.nii' in nii_fn):
                nii_fn += '.nii'

            elif not ('.gz' in nii_fn):
                nii_fn += '.gz'

            else:
                raise FileNotFoundError(f"File not found: {nii_fn_orig}")

        else:
            break

    return newnii


def get_nii_prefix(nii_path):
    """Strip .nii or .nii.gz from end of filename as necessary; return stripped string."""

    nii_prefix = str(nii_path).rsplit('.nii')[0]    # handles .nii or .nii.gz or none

    # # if a string is passed, make a Path object
    # if isinstance(nii_path, str):
    #     nii_path = Path(nii_path)

    # if nii_path.suffix == '.nii':
    #     nii_prefix = str(nii_path)[:-4]

    # elif nii_path.suffixes == ['.nii', '.gz']:
    #     nii_prefix = str(nii_path)[:-7]

    # else:
    #     nii_prefix = str(nii_path)

    return nii_prefix


def make_nii(new_img, new_aff=None, new_hdr=None):
    """Create a new nii object; use orig affine, units by default.

    Some data types will be converted, as follows:
    - bool -> uint8 (aka unsigned char)
    - float64 -> float32

    Returns a NIfTI object from nib.Nifti1Image(). Can be used to write out a numpy array to a
    NIfTI file using, e.g.:

    make_nii(np_array, new_aff=nii.affine).to_filename('new_file.nii.gz')

    """

    # Look after output dtype:
    # - uint8 is good for bools and small ints like segmentations with a few labels
    # - float32 should be precise enough for floats
    # - the following code is safe because .astype makes a copy of the passed object
    set_dtype_float32 = False

    if new_img.dtype == 'bool':
        # bool not supported, use uint8: 0 -> 255; aka C unsigned char
        new_img = new_img.astype(np.uint8)

    elif new_img.dtype == 'float64':
        # float64 is default, but 32 is plenty for our purposes, and more compatible
        set_dtype_float32 = True


    if new_hdr is None:
        new_nii = nib.Nifti1Image(new_img, new_aff)
        new_nii.header.set_xyzt_units('mm', 'sec')      # or nii.get_xyzt_units()
        if set_dtype_float32:
            new_nii.set_data_dtype(np.float32)          # dtype that will be used when image is saved
        new_nii.set_qform(None, code=1)
        new_nii.set_sform(None, code=1)

    else:
        # if new_aff is None, affine will come from the header
        new_nii = nib.Nifti1Image(new_img, new_aff, header=new_hdr)
        new_nii.set_sform(None, code=1)     # default was 2, and qform zeroed out and code=0

    new_nii.header.structarr['descrip'] = b'CAN-BIND dMRI'

    return new_nii


def rm_nii(nii_fn):
    """Remove nii file using path that may be prefix or include .nii or .nii.gz"""

    nii_pfx = get_nii_prefix(nii_fn)

    nii_files = glob(f'{nii_pfx}.nii*')

    assert len(nii_files) > 0, "rm_nii: Expected at least 1 matching file"

    for f in nii_files:
        os.remove(f)


def mv_nii(nii_fn1, nii_fn2):
    """Move nii file using path that may be prefix or include .nii or .nii.gz"""

    nii1_pfx = get_nii_prefix(nii_fn1)
    nii1_file = glob(f'{nii1_pfx}.nii*')
    assert len(nii1_file) == 1, f"mv_nii: Expected 1 matching file for nii_fn1: {nii_fn1}"
    nii1_file = nii1_file[0]

    # get suffix(es)
    n = len(nii1_file) - len(nii1_pfx)
    nii1_sfx = nii1_file[-n:]

    nii2_pfx = get_nii_prefix(nii_fn2)

    os.rename(nii1_file, nii2_pfx + nii1_sfx)


def vec_to_img(vec, mask):
    """Populate ndarray mask with values; e.g. labels, MSE values.

    This is the inverse of numpy slicing operations like:
      foo = arr_3d[mask]
      bar = arr_4d[mask, :]
    """

    # Create 3D image and input values
    img = np.zeros(mask.shape, dtype=vec.dtype)
    img[mask] = vec

    return img


def math_op_nii(img_fn, op, shenv=None):
    """Apply an in-place mathematical operation to a nifti image such as thresholding or masking.
    Pass the file-name of an existing nii, an operation for fslmaths, and optionally the shell ENV.

    Examples:
    math_op_nii(img_fn, '-thr 0')
    math_op_nii(img_fn, f'-mul {mask_fn}')
    """

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"

    # use fslmaths to perform the operation
    sp.run('$FSLDIR/bin/fslmaths'
          f' {img_fn} {op} {img_fn}',
           shell=True, check=True, env=shenv)


def stat_op_nii(img_fn, op, mask=None, shenv=None):
    """Retrieve a statistic from a nifti image such as median or robust intensity range.
    Pass the file-name of an existing nii, an operation for fslstats, and optionally
    a mask filename and the shell ENV.

    Examples:
    stat_op_nii(img_fn, '-P 50')
    """

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"

    if mask is None:
        maskopt = ''
    else:
        maskopt = f'-k {mask}'

    # call fslstats
    stat_out = sp.run('$FSLDIR/bin/fslstats'
                     f' {img_fn} {maskopt} {op}',
                      capture_output=True, text=True,
                      shell=True, check=True, env=shenv)

    return float(stat_out.stdout)


def gauss_smooth(img, HWHM_mm=0.75):
    """Smooth ndimage isotropically

    - by default, FWHM = 1.5 mm (sigma ~ 0.64 mm)
    - caution: smoothes into backround area beyond brain, requires remasking after

    returns smoothed ndimage
    """

    # convert mm -> voxels
    voxdims = nii.header.get_zooms()
    vdim_3d = np.mean(voxdims[0:3]).round(3)  # average vox dim in mm

    HWHM_vox = HWHM_mm/vdim_3d

    # previous default: HWHM_vox=0.5

    # FWHM = 2.355 x sigma
    sigma = HWHM_vox*2/2.355

    sm_img = ndi.gaussian_filter(img, sigma, mode='nearest')    # mode says how the input array is
                                                                # extended beyond a border

    return sm_img


def morph_ops_nii(nii_fn, op, rad=5.0, odt='float', shenv=None):
    """Apply morphological operations in-place to a NIfTI volume using fslmaths.
    Operations supported:
    - opening
    - closing

    Radius is in mm. Set odt=char for masks.
    """

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"

    # prepare operation
    if op == 'opening':
        op_args = '-ero -dilD'
    elif op == 'closing':
        op_args = '-dilD -ero'


    if rad is None:
        # radius
        # n.b. to fully cover voxels across a double gap, need radius of ~ 1.659*v
        pixdims = nib.load(nii_fn).header.get_zooms()
        v = np.max(pixdims[0:3])
        rad = 1.659*v


    sp.run('$FSLDIR/bin/fslmaths'
          f' {nii_fn}'
          f' -kernel sphere {rad} {op_args}'
          f' {nii_fn} -odt {odt}',
           shell=True, check=True, env=shenv)


def mask_morphops(mask, closing_first=True, fill_holes=True, clust_mass_img=None, rad1=1, rad2=2):
    """Morphological operations to clean up masks (trim islands/bridges/peninsulas, fill holes),
    using the following general operations by default:
    0. binary closing
    1. hole-filling
    2. erosion
    3. labelling, choose LCC
    4. dilation

    If closing_first is true, this will first do a binary closing operation (dilate, erode),
      which can help with mask gaps in areas around the cerebellum.

    If clust_mass_img is given, the structuring element will be a square-connectivity cross with
      diameter = 3, otherwise balls with the given radii.

    The radii are in adjusted units, equivalent to 2.5 mm voxels; rad=1 should close a ~ 5 mm gap,
      rad=1.5 should close a ~ 7.5 mm gap. However, this depends on the course integer rounding of
      the adjusted radii. I have seen gaps as big as 6 mm that need closing in brain masks.
    """

    # Use voxel dimensions to adjust for higher-res data (e.g. HCP 1.5 mm iso instead of 2.5)
    voxdims = nii.header.get_zooms()
    vdim_3d = np.mean(voxdims[0:3]).round(3)  # average vox dim in mm

    if clust_mass_img is None:
        print(" "*4 + f"mask_morphops: vdim_3d={vdim_3d:.3f}, using ball")

        # for brain masks, use ball with (adjusted) radius given
        #   larger structure element has greater effect
        #   radius 2 generates a ball with diameter 5 and so on
        r1 = np.round(rad1*2.5/vdim_3d).astype(int)  # voxel units equiv to 2.5 mm voxels
        strelem1 = ball(r1)

        r2 = np.round(rad2*2.5/vdim_3d).astype(int)  # ~ 5 mm by default
        strelem2 = ball(r2)

    else:
        # for clusters, use default structure element: 3D 'plus' with diameter 3, no diagonals
        strelem1 = ndi.generate_binary_structure(3, 1)  # view with .astype(np.int)
        strelem2 = strelem1


    # 0. First close gaps if requested
    if closing_first:
        mask = ndi.binary_closing(mask, structure=strelem1)

    # 0. Fill holes if requested (uses square connectivity)
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)

    # 1. Erosion (beginning of opening = erosion/dilation)
    mask = ndi.binary_erosion(mask, structure=strelem2)

    # 2. Label image and to find connected components (uses square connectivity)
    labs_img, n_feat = ndi.label(mask)    # labs_img is ndarray with integer labels

    if clust_mass_img is None:
        # take the largest connected component (LCC)
        lcc_lab = stats.mode(labs_img[labs_img > 0], axis=None).mode[0]  # mode of non-zero labels

        mask = (labs_img == lcc_lab)

    else:
        # for clusters with mass, take the highest mass cluster
        lab_masses = np.zeros([n_feat+1])

        for l in range(1, n_feat+1):
            cmask = (labs_img == l)
            mass = np.sum(clust_mass_img[cmask])

            lab_masses[l] = mass

        mask = (labs_img == np.where(lab_masses == lab_masses.max())[0][0])

    # 3. Dilation (finish opening)
    mask = ndi.binary_dilation(mask, structure=strelem2)

    # n.b. it's tempting to do one more dilation here, but testing shows it results
    #      in a ring of noise on the FA

    return mask


def sigmoid_xfm(data, normp=None, normv=None):
    """Transform input data using a sigmoid function to suppress long histogram tails.

    This is a modified sigmoid function so that it's linear through the origin,
    but flattens toward 1 at high values:

        y = 2/(1+exp(-2x)) - 1

    It's close to 1:1 slope < 0.5 (passes through (0.25, 0.25), but then goes
    through (1.0, 0.75), (1.5, 0.9), (2.0, 0.96) etc

    Accepts a numpy array, returns transformed array
    """

    if normp is not None:
        # normalize by this percentile of the signal
        q = np.percentile(data, normp)
        data = data/q

    elif normv is not None:
        # normalize by this absolute value
        data = data/normv

    xfm_data = 2/(1 + np.exp(-2*data)) - 1

    return xfm_data


def remove_bias(dwi_data, threshmask, noise_resid_fns=None, affine=None, shenv=None):
    """Estimate bias field by running FSL FAST on mean T2-w image.
    Remove bias field from (noise and ringing reduced) DWI data.

    Return bias-corrected data and the file name.
    """

    if affine is None:
        affine = nii.affine

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"


    # Need to create a T2w image from non-MC'd input data for fast to work on
    # Notes:
    # - interpolation steps always introduce negative values, but raw has min=0
    # - also tried np.min() volume to try to take advantage of artifacts, but didn't seem to help
    # - tried adding dw_mean as an extra channel to fast, but seemed to make anatomical details
    #   _more_ distinct in the bias map, so abandoned that strategy; increasing the smoothing
    #   using -l 40 in the command call helps somewhat

    t2w_img = np.mean(dwi_data[..., t2w_vols], axis=3)
    t2w_img_fn = f'{tmpd}/biasx_t2w_img_for_fast.nii'
    make_nii(t2w_img, new_aff=affine).to_filename(t2w_img_fn)

    # Use FAST to estimate bias
    t2w_fastseg_pp = f'{tmpd}/t2w_biasx'

    print(" "*4 + "remove_bias: calling fast")
    if verbose:
        vstr = '-v'
    else:
        vstr = ''

    sp.run('$FSLDIR/bin/fast'
          f' --type=2 --class=4 --nopve -b {vstr}'
          f' -o {t2w_fastseg_pp} {t2w_img_fn}',
           shell=True, check=True, env=shenv)

    biasfield_fn = t2w_fastseg_pp + '_bias.nii'
    bias_field = nib.load(biasfield_fn).get_fdata()     # ~ 0.5 -- 1.5 float data

    # Report rough bias magnitudes
    bf = bias_field[threshmask]
    x = np.percentile(bf, 2)
    y = np.percentile(bf, 98)
    print(" "*4 + f"remove_bias: 2%–98% bias: {x:0.2f}–{y:0.2f}")


    # Bias-correct 4D data
    dwi_biascorr = dwi_data/bias_field[..., None]

    dwi_biascorr_fn = f'{tmpd}/dwi_biascorr.nii'
    make_nii(dwi_biascorr, new_aff=affine).to_filename(dwi_biascorr_fn)


    # Also bias-correct noise and residual maps in-place
    for fn in [noise_resid_fns[i] for i in (0,1,3)]:
        math_op_nii(fn, f'-div {biasfield_fn}', shenv=shenv)


    # Clean up input file for fast
    os.remove(t2w_img_fn)

    return dwi_biascorr, dwi_biascorr_fn


def calc_entropy(dwi_data, mask, affine=None):
    """Calculate entropy from dwi_data using the power spectral method.

    Returns image with entropy values within mask, 0 outside.
    """

    if affine is None:
        affine = nii.affine

    # N.B. Using neurokit, examined shannon, spectral, and svd entropy types
    #      also tested entr from scipy.special; very similar to shannon from neurokit
    # shannon_entropy: not useful; no contrast btw brain and background
    # spectral_entropy: good! threshold of < 0.5 will eliminate eyeballs, but also some CSF and WM
    #       nzmean    P2   P50   P98
    #   csf: 0.168 0.016 0.059 1.174
    #    gm: 0.106 0.018 0.055 0.683
    #    wm: 0.213 0.036 0.166 0.778
    #    bg: 0.996 0.064 0.925 2.577
    # svd_entropy: maybe good too. nzmean P2 P50 P98:
    #   csf: 0.409 0.206 0.351 0.869
    #    gm: 0.376 0.215 0.343 0.772
    #    wm: 0.515 0.284 0.514 0.800
    #    bg: 0.782 0.360 0.820 0.978

    # conclusion: svd & spec entropy are very similar, but svd gives a few more false-positives in the brain.
    #   -- choose spectral_entropy with threshold @ P98 for WM

    # Internal calc of frequency domain / (power) spectral entropy. Basic strategy:
    # - compute PSD
    # - normalize it to obtain probability-like values (integral=1)
    # - calculate entropy from the probs
    # See this answer: https://dsp.stackexchange.com/a/23697/33787
    #     or this one: https://stackoverflow.com/a/30465336/1329892
    # or the neurokit code (https://github.com/neuropsychology/NeuroKit.py)

    # previously used slice-at-a-time loop, but was less efficient than vectorized
    #   version by a wide margin
    # complexity = nk.complexity(s, sampling_rate=1/TR,
    #                            sampen=False, multiscale=False, correlation=False,
    #                            higushi=False, petrosian=False, fisher=False,
    #                            hurst=False, dfa=False)
    #  shannon_entropy[i,j,k] = complexity['Entropy_Shannon']
    #  spectral_entropy[i,j,k] = complexity['Entropy_Spectral']
    #  svd_entropy[i,j,k] = complexity['Entropy_SVD']

    ent_ims = []
    for b in nzbvals_uniq:
        # entropy per shell; using all at once on multi-shell gives somthing like T2-w contrast
        vols_tf = (bvals == b)

        # pre-allocate array
        ent_ps = np.zeros(dwi_data.shape[:-1])

        # Vectorized calculation within mask only
        #   create voxels x time array to make it easy to apply flattened mask
        n_vox = np.product(ent_ps.shape)
        dw_vxt = dwi_data[..., vols_tf].reshape(n_vox, -1)[mask.ravel(), :]

        # N.B. after reshape but before mask, dw_vxt[242138, :] is the sames as
        #   dwi_data[48, 48, 26, vols_tf]
        # this is because the last counter rolls first:
        # dw_vxt[0, :] == dwi_data[0, 0, 0, vols_tf]
        # dw_vxt[1, :] == dwi_data[0, 0, 1, vols_tf]
        # so 242138 = 48*96*52 + 48*52 + 26
        # This can be found using np.unravel_index(242138, (96,96,52)) -> (48, 48, 26)

        # Calculate PSD, neglecting normalization
        PSD = np.abs(np.fft.rfft(dw_vxt))**2   # PSD.shape -> (107873, 16); i.e. nvox in mask x 1/2 directions + 1

        # N.B. calculating the PSD with normalization leads to PS-Entropy with no contrast at all. E.g.:
        # dw_vxt_normed = dw_vxt - dw_vxt.mean(axis=1)[:,None]
        # PSD = np.abs(np.fft.rfft(dw_vxt_normed))**2   # PSD.shape -> (107873, 16)
        # - In other words, the PS-Entropy depends heavily on the mean signal, which is not so good
        # - Could also use `from scipy import signal; signal.spectral.periodogram(s, detrend=False)`
        #   values are slightly different, but have the same relationship

        # Convert to probabilities using sum across directions as column vector
        # Note norm_f may have 0's, resulting in a 0/0 situation in the division
        #   also results in -inf values from the log2
        #   these should be on the boundary of the mask, so it would make sense to
        #   set the entropy result to 0 for these undefined spots
        norm_f = PSD.sum(axis=1)[:, None]
        prob_D = np.divide(PSD, norm_f, out=np.zeros_like(PSD), where=(norm_f!=0))
        log_prob = np.log2(prob_D, out=np.zeros_like(prob_D), where=(prob_D!=0))

        # Calculate Spectral entropy
        ent_ps[mask] = -1*np.sum(prob_D*log_prob, axis=1)

        # PS-Entropy
        #       nzmean    P2   P50   P98
        #   csf: 0.168 0.016 0.059 1.174
        #    gm: 0.106 0.018 0.055 0.683
        #    wm: 0.213 0.036 0.166 0.778
        #    bg: 0.995 0.062 0.924 2.577

        ent_ims.append(ent_ps)

    ent_total = np.sum(ent_ims, axis=0)     # back to 3D

    ent_fn = f'{tmpd}/entropy.nii'
    make_nii(ent_total, new_aff=affine).to_filename(ent_fn)

    return ent_total, ent_fn


def fit_tensor(dwi_fn, mask_fn, bval_fn=None, bvec_fn=None, sse=True, wls=True, \
               save_tensor=True, kurtdir=False, single_b=None, affine=None, shenv=None):
    """Fit diffusion tensor to preprocessed DWI data.

    Outputs files related to the tensor and its metrics next to the input file.

    Returns prefix of output file paths.
    """

    if affine is None:
        affine = nii.affine

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"

    if verbose:
        stdout = None
    else:
        stdout = sp.DEVNULL

    dwi_fn_pfx = get_nii_prefix(dwi_fn)

    if bval_fn is None:
        bval_fn = f'{dwi_fn_pfx}.bval'

    if bvec_fn is None:
        bvec_fn = f'{dwi_fn_pfx}.bvec'

    # output prefix, unless changed in options below
    outdir = dirname(dwi_fn)
    tensor_pp = f'{outdir}/dt'

    # tensor fitting options
    dtifit_flags = []

    if sse:
        dtifit_flags.append('--sse')

    if wls:
        dtifit_flags.append('--wls')
        tensor_pp = f'{outdir}/dtw'

    if save_tensor:
        dtifit_flags.append('--save_tensor')

    if kurtdir:
        dtifit_flags.append('--kurtdir')
        tensor_pp = f'{outdir}/dtk'

    if single_b is not None:
        dwi_ss_fn = f'{dwi_fn_pfx}_ssb{single_b}'

        sp.run('$FSLDIR/bin/select_dwi_vols'
              f' {dwi_fn} {bval_fn} {dwi_ss_fn} {single_b}'
              f' -b 0 -obv {bvec_fn}',
              shell=True, check=True, env=shenv, stdout=stdout)

        dwi_fn = dwi_ss_fn
        bval_fn = f'{dwi_ss_fn}.bval'
        bvec_fn = f'{dwi_ss_fn}.bvec'
        tensor_pp += f'_b{single_b}'


    # fit tensor
    sp.run(f'$FSLDIR/bin/dtifit ' + ' '.join(dtifit_flags) +
           f' --data={dwi_fn} --out={tensor_pp} --mask={mask_fn}'
           f' --bvals={bval_fn} --bvecs={bvec_fn}',
            shell=True, check=True, env=shenv, stdout=stdout)


    # Convert diffusivity values to um^2/ms for convenience of later processing
    ims = ['L1', 'L2', 'L3', 'MD']
    if save_tensor:
        ims.append('tensor')

    for im in ims:
        img_fn = tensor_pp + '_' + im
        math_op_nii(img_fn, '-mul 1000', shenv=shenv)


    if save_tensor:
        # Explicity name the tensor to be  upper- and lower-triangular, since the FSL tensor
        #   component order is not the NIfTI standard.
        # NIfTI dictates the tensor in order (xx, yx, yy, zx, zy, zz) (i.e. lower)
        # FSL outputs the tensor in order (xx, xy, xz, yy, yz, zz) (i.e. upper)
        # see http://groups.google.com/group/dtitk/web/Interoperability+with+major+DTI+tools
        tens_up_fn = tensor_pp + '_tensor_upper.nii'
        os.rename(tensor_pp + '_tensor.nii', tens_up_fn)

        tens_up = nib.load(tens_up_fn).get_fdata()  # shape -> (96, 96, 58, 6)
        tens_down = tens_up[..., [0, 1, 3, 2, 4, 5]]
        make_nii(tens_down, new_aff=affine).to_filename(tens_up_fn.replace('upper', 'lower'))

    return tensor_pp


def rough_bbgmasking(dwi_data, fine_pass=False, affine=None, shenv=None):
    """Create rough brain mask from mixed-contrast DWI data using bet, with some
    morpholoigal operations.

    returns:
    - mask-rough_brainbg: output of bet + morph-ops and thresh > 0.001 (and filename)
    - mask-bimod_brainbg: within rough_brainbg mask, threshold_minimum is applied to mixed_dwmean,
                              then largest connected component is kept
    """

    if affine is None:
        affine = nii.affine

    # Create mean image (T2w/3+DWI; mixed contrast) for masking target -- this is better than
    #   straight T2 because it tends to tamp down the extreme CSF signal
    t2w_mean = np.mean(dwi_data[..., t2w_vols], axis=3)
    dw1_mean = np.mean(dwi_data[..., dw1_vols], axis=3)
    mixed_dwmean = 1/3.*t2w_mean + dw1_mean

    # In HCP data, the mixed_dwmean image had worse SNR and a lot of 'neck', so a lot of
    #   low (cheek?) non-brain tissue was kept in the bet mask; using T2w solved the issue
    # So, on the second pass (fine_pass), use T2w for bet input
    # Regardless, mixed_dwmean image (not file) is needed for threshmask below

    if fine_pass:
        t2w_mean_fn = f'{tmpd}/raw_t2wmean.nii'
        make_nii(t2w_mean, new_aff=affine).to_filename(t2w_mean_fn)

        # Rough brain mask file names generated by bet
        bet_input_fn = t2w_mean_fn
        bet_roughbrain_fn = f'{tmpd}/raw_t2wmean_brain'
        bet_roughmask_fn = bet_roughbrain_fn + '_mask'

    else:
        mixed_dwmean_fn = f'{tmpd}/mixed_dwmean.nii'
        make_nii(mixed_dwmean, new_aff=affine).to_filename(mixed_dwmean_fn)

        bet_input_fn = mixed_dwmean_fn
        bet_roughbrain_fn = f'{tmpd}/mixed_dwmean_brain'
        bet_roughmask_fn = bet_roughbrain_fn + '_mask'

    # Running bet with f=0.1 gives generous mask, but low risk of clipping brain tissue
    print(" "*4 + "rough_bbgmasking: calling bet")
    if verbose:
        vstr = '-v'
    else:
        vstr = ''

    sp.run('$FSLDIR/bin/bet'
          f' {bet_input_fn} {bet_roughbrain_fn}'
          f' -f 0.1 -m -R {vstr}',
           shell=True, check=True, env=shenv)

    rm_nii(bet_roughbrain_fn)

    # rename mask to generalize
    roughmask_fn = f'{tmpd}/mask-rough_brainbg'
    mv_nii(bet_roughmask_fn, roughmask_fn)


    # Run morphological opening to try to get rid of protrusions (e.g. eyeball bits)
    print(" "*4 + "rough_bbgmasking: morphological opening on rough mask")

    morph_ops_nii(roughmask_fn, op='opening', rad=5.0, odt='char', shenv=shenv)

    # Load mask as bool
    roughmask_nii = check_load_nii(roughmask_fn)
    roughmask = roughmask_nii.get_fdata().astype('bool')  # 1/0 binary data -> True/False
    roughmask_fn = roughmask_nii.get_filename()


    # It's good to exclude approx. zero voxels, since they cause problems for dtifit;
    # Low signal values in b=0 images can cause dtifit to fail with error:
    #   "solve(): solution not found" when using --wls, so they must be excluded.
    # Therefore, check for all near-zero signals within roughmask and touch up the mask accordingly
    # These occur outside the brain, mostly on the top and bottom slice
    # A way of fixing the dtifit error is touching up the mask with fslmaths like this:
    #   fslmaths t2w_roughbrain -thr 0.001 -bin -mas t2w_roughbrain_mask newmask
    # See this [mailing-list post](https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=FSL;e1860e73.1610)

    # Note this can create small holes in the roughmask where signal dips below 0 from
    #   ringing artifacts, such as btw the brain and skull; such voxels may be included in
    #   the bimod_mask created below. In practice that's OK because the roughmask is used
    #   for tensor/MSE calculation in later processing.

    # Check to make sure the data has large enough intensity to use an absolute tolerance
    highval = np.percentile(dwi_data, 98)
    if highval < 100:
        print(f"Warning: dwi_data scaling is very low; 98th percentile is {highval}")

    # Ensure all voxels have signal of at least 0.001 for some time pts
    thresh_gt0 = 0.001
    t2w_zeros = np.all((dwi_data[..., t2w_vols] < thresh_gt0), axis=3)  # 3D volume of np.bool_
    dw_zeros = np.all((dwi_data[..., ~t2w_vols] < thresh_gt0), axis=3)

    nz_mask = (~t2w_zeros) & (~dw_zeros)  # non-zero voxels = 1/True

    roughmask = roughmask & nz_mask
    make_nii(roughmask, new_aff=affine).to_filename(roughmask_fn)

    mask_perc = 100*roughmask.sum()/np.prod(roughmask.shape)
    print(" "*4 + f"rough_bbgmasking: rough mask is {mask_perc:0.1f}% of total volume (~ 17–32% is good)")


    # Also make a mask by thresholding a mixed contrast image
    #   - mixed contrast image is fairly bimodal, so this works pretty well
    #   - threshold_minimum tends to give a lower (safer, more useful) value than
    #     threshold_otsu, but is more susceptible to very high-signal outliers
    #     -> the minimum algorithm takes a histogram of the image and smooths it repeatedly
    #        until there are only two peaks; then thresh is the minimum between the peaks
    #     -> the otsu algorithm minimizes variance within clusters (should give same as kmeans)
    #   - a bit of gaussian smoothing should tamp down the outliers, and can exclude
    #     the top and bottom 1% of values as well
    print(" "*4 + "rough_bbgmasking: generating bimodal threshold mask from mixed DW contrast image")

    mixed_dwmean_smth = gauss_smooth(mixed_dwmean)

    p1 = np.percentile(mixed_dwmean_smth[roughmask], 1)
    p99 = np.percentile(mixed_dwmean_smth[roughmask], 99)

    mm = roughmask & (mixed_dwmean_smth > p1) & (mixed_dwmean_smth < p99)

    thresh_min = threshold_minimum(mixed_dwmean_smth[mm])
    bimod_mask = roughmask & (mixed_dwmean_smth > thresh_min)

    # Use mask_morphops to smooth protrusions, fill holes, and take the
    #   largest connected component with square connectivity
    # steps: closing, hole-fill, erode, LCC, dilation
    # output mask stays as bool type
    bimod_mask = mask_morphops(bimod_mask, rad2=1)

    bimodmask_fn = f'{tmpd}/mask-bimod_brainbg.nii'
    make_nii(bimod_mask, new_aff=affine).to_filename(bimodmask_fn)

    mask_perc = 100*bimod_mask.sum()/np.prod(bimod_mask.shape)
    print(" "*4 + f"rough_bbgmasking: bimod mask is {mask_perc:0.1f}% of total volume (~ 15–26% is good)")

    return roughmask, roughmask_fn, bimod_mask, bimodmask_fn


def rough_seg_wm(dwi_data_fn, affine=None, shenv=None):
    """Create a rough WM segementation from bias-corrected dMRI data; this is primarily to
    normalize the signal level of the DW+T2W images across scanner vendors.
    """

    if affine is None:
        affine = nii.affine

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"
    assert 'DMRIDIR' in shenv, "DMRIDIR not set in envirionment"

    # calculate RISH images
    sp.run(f'$DMRIDIR/dmri_rishes {dwi_data_fn} 2',
           shell=True, check=True, env=shenv)

    # segment l=2 RISH into likely WM and everything else
    rish_l2_fn = f'{dwi_data_fn}-rish_l2'

    fast_fn = f'{dwi_data_fn}-fast_n3_c1_rish2xfm'

    # run rish_l2 through a sigmoid transform so the top end tail of the distribution is not so long
    rish2 = check_load_nii(rish_l2_fn).get_fdata()
    rish2_xfm = 1000*sigmoid_xfm(rish2, normp=99)

    rish2_xfm_fn = f'{rish_l2_fn}_xfm'
    make_nii(rish2_xfm, new_aff=affine).to_filename(rish2_xfm_fn)

    # use fast for clustering
    print(" "*4 + "rough_seg_wm: calling fast")
    if verbose:
        vstr = '-v'
    else:
        vstr = ''

    n_classes = 3
    sp.run('$FSLDIR/bin/fast'
          f' --class={n_classes} --nobias {vstr}'
          f' -o {fast_fn} {rish2_xfm_fn}',
           shell=True, check=True, env=shenv)

    # Note: FAST outputs
    # - pve_X are 0–1 probability maps for each cluster/tissue type
    # - pveseg image segments tissues based on highest probability tissue type
    #   values of pveseg are i+1, since 0 is "outside the brainmask"
    # - number order determined by mean intensity per tissue (darkest to brightest)
    # - seg and seg_X images are hard binary seg images that do not necessarily
    #   correspond to the pveseg ones; these do not seem to be very useful; pveseg > seg
    # - when using multi-channel segmentation, don't use -t; order must be determined

    pveseg = check_load_nii(fast_fn + '_pveseg').get_fdata()
    rish2 = check_load_nii(rish_l2_fn).get_fdata()

    # seg-mask with highest median value is WM
    rish2_medians = []
    for i in range(n_classes):
        # mask_fn = fast_fn + f'_seg_{i}'
        # mask_mdn = stat_op_nii(rish_l2_fn, '-p 50', mask=mask_fn, shenv=shenv)
        mdn = np.median(rish2[pveseg == i+1])
        rish2_medians.append(mdn)

    i_wm = rish2_medians.index(max(rish2_medians))

    wm_pve_fn = f'{fast_fn}_pve_{i_wm}'
    wm_pve = check_load_nii(wm_pve_fn).get_fdata()

    os.rename(f'{wm_pve_fn}.nii', f'{fast_fn}_WMpve.nii')


    # clean up
    for fn in glob(dwi_data_fn + '-rish_l[0-9].*') \
            + glob(rish2_xfm_fn + '.*') \
            + glob(dwi_data_fn + '-sharm[0-9].*') \
            + glob(fast_fn + '_seg*.*') \
            + glob(fast_fn + '_pve*.*') \
            + glob(fast_fn + '_mixeltype*.*'):
        os.remove(fn)

    wm_mask = (wm_pve > 0.9)
    mask_perc = 100*wm_mask.sum()/np.prod(wm_mask.shape)
    print(" "*4 + f"rough_seg_wm: WM mask is {mask_perc:0.1f}% of total volume (~ 3–6% is good)")

    return wm_pve


def fine_bbgmasking(dwi_data, bvals_path, affine=None, shenv=None, db_seg=False):
    """Segment DWI data into brain and background based on the following metrics:
    - RISH l=0 (half-sigmoidal transformed)
    - RISH l=2 (half-sigmoidal transformed)
    - MSE of a tensor fit

    This function tends to exclude some superficial areas of CSF and background that give poor model fits.

    Pass bias-corrected data for dwi_data.

    Returns volume with 0 for background and 1 for brain.
    """

    if affine is None:
        affine = nii.affine

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"
    assert 'DMRIDIR' in shenv, "DMRIDIR not set in envirionment"

    # Notes on segmentation testing:
    # - sqrt_rish0 (same as dw1000_mean) with n=3 gives BG+CSF vs brain tissues mask,
    #   though you have to watch out for the globus pallidus getting lost; eyeball signal is
    #   tamped down too, so that's very useful
    # - with n=2: globus pallidus and lots of CSF is out; too harsh
    # - 3rt_rish0: fast_n3_c1_3rtrish0_pveseg is even better for globus pallidus
    # - rish2 with n=3 also gives decent BG+CSF vs brain tissues mask

    # Winning combo, without using MSE: fast_n3_c2_sigmdrish0_sigmdrish2_pveseg

    # best of the rest for 3 classes (brain/bg):
    # 1. fast_n3_c1_3rtrish0_pveseg (a bit more restrictive than above)
    # 2.           _sigmdrish0_pveseg
    # 3.           _sqrtrish0_pveseg
    # 4.           _rish2_pveseg
    # 5. fast_n3_c2_sigmdrish0_rish2_pveseg (gaps at temporal or more cruft, depending on threshold)
    #              _3rtrish0_rish2
    #              _sqrtrish0_rish2

    # best of the rest for 4 classes:
    # 1. fast_n4_c3_t2w_sqrtrish0_rish2_pveseg -- a bit finer WM; also has some cruft, and eyeballs
    # 2. fast_n4_c3_t2w_sigmdrish0_sigmdrish2_pveseg -- for brain/bg, incl CSF has cruft, but excl is too restrictive
    # 2. fast_n4_c3_sigmdt2w_sqrtrish0_rish2_pveseg -- as above 2; still good very good choices at thresh of 2.5 though
    # 2. fast_n4_c2_t2w_sqrtrish0_pveseg
    # 3. fast_n4_c2_t2w_3rtrish0_pveseg
    # 3. fast_n4_c2_t2w_4rtrish0_pveseg
    # 5. fast_n4_c2_t2w_rish0_pveseg

    # sigmoidal transform of rish-0 to reduce the effect of the susceptibility artifact
    # modify the sigmoid so that it's lienar through the origin, but flattens at high values:
    # y = 2/(1+exp(-2x)) - 1
    # ^^^ close to linear < 0.5 (passes through (0.25, 0.25), but then goes through (1.0, 0.75), (1.5, 0.9), (2.0, 0.96) etc
    # so put the median WM value of rish0 right at 0.75 before the transform
    # could also consider basing this off the 99.9% value in the image
    # fslstats dwi_eddycorr-rish_l0 -k fast_n3_c1_rish2_WMseg -p 50
    # 306230.375000
    # 306230/0.75 = 408306
    # fslmaths dwi_eddycorr-rish_l0 -div 408306 dwi_eddycorr-rish_l0_wm075
    # 3dcalc -a dwi_eddycorr-rish_l0_wm075.nii -expr '(2/(1+exp(-2*a))-1)*2000' -prefix dwi_eddycorr-rish_l0_sigmoid.nii
    #
    # transform l2 as well
    # fslstats dwi_eddycorr-rish_l2 -p 99.
    # 13066
    # fslmaths dwi_eddycorr-rish_l2 -div 13066 dwi_eddycorr-rish_l2_p99eq1
    # 3dcalc -a dwi_eddycorr-rish_l2_p99eq1.nii -expr '(2/(1+exp(-2*a))-1)*2000' -prefix dwi_eddycorr-rish_l2_sigmoid.nii
    #
    # and T2-w; why not?
    # fslstats t2w_mean -p 99.
    # 1017
    # fslmaths t2w_mean -div 1017 t2w_mean_p99eq1
    # 3dcalc -a t2w_mean_p99eq1.nii -expr '(2/(1+exp(-2*a))-1)*2000' -prefix t2w_mean_sigmoid.nii

    # should SSE mask be involved here?
    #   yes, I love SSE; but convert to MSE first for consistency, and run through the sigmoid
    # $FSLDIR/bin/select_dwi_vols dwi_eddycorr dwi_eddycorr.bval dwi_eddycorr_b1000 1000 -b 0 -obv dwi_eddycorr.bvec
    # $FSLDIR/bin/dtifit --sse --wls --data=dwi_eddycorr_b1000 --out=wdt_b1000 --mask=mask-rough_brain --bvals=dwi_eddycorr_b1000.bval --bvecs=dwi_eddycorr_b1000.bvec
    # # Calculate mean-squared error: sse/edof
    #   N = len(t2w_vols)         # SSE includes all b=0, and S_0 is an estimated parameter,
    #   mse = sse/(N - 6 - 1)     #   so 7 params and N = all volumes
    #                             #   but if SSE comes from b=1000 shell only, N=39 here
    # 3dcalc -a wdt_b1000_sse.nii -expr 'a/(39-6-1)' -prefix wdt_b1000_mse.nii
    # 3dcalc -a wdt_b1000_mse.nii -expr '(2/(1+exp(-2*a))-1)' -prefix wdt_b1000_mse_sigmoid.nii
    # or
    # fslstats wdt_b1000_mse.nii -p 99.
    # 0.304
    # fslmaths wdt_b1000_mse -div 0.304 wdt_b1000_mse_p99eq1
    # 3dcalc -a wdt_b1000_mse_p99eq1.nii -expr '(2/(1+exp(-2*a))-1)' -prefix wdt_b1000_mse_p99sigmoid.nii
    # - fast_n3_c1_mse-p99sigmd_pveseg is too ruthless
    # - running fast with n=3 on straight mse is good, maybe fast_n3_c1_mse-sigmd_pveseg will be better
    #   in combo with other maps, because less skull
    #   testing this with fast_n3_c2_sigmdrish0_sigmdrish2_pveseg:
    #     + using sigmoid-MSE much messier than straight MSE; use regular MSE
    # - take the cluster with miniumum MSE:
    #   $FSLDIR/bin/fslmaths fast_n3_c3_sigmdrish0_sigmdrish2_mse_pveseg -uthr 1.5 fast_n3_c3_sigmdrish0_sigmdrish2_mse_brainmask
    # - this reintroduced the eyeballs, made other small changes around the periphery; moving ahead with this,
    #   because I'm expecting that introducing the MSE metric should make the segmentation more stable across subjects/scanners


    # OK, above says how to segment into brain/BG reliably; now how best to segment this reduced
    #   FOV into 3 tissue types?
    # - rish_l2 has the most WM/GM contrast, but not much CSF contrast
    # - try rish_l2 + T2w? OR incorporate RD image? or L1?
    # - Am I OK with having CSF and BG lumped together, maybe? Then can use Rish0 and Rish2 again.
    #   then further segment the CSF/BG mask if necessary into high/low
    #
    # 6 Step segmentation process:
    #   1. Generate rough mask using bet; this strips some of the eyeballs, but not all
    #   2. Segment within roughmask using sigm_rish_l0, sigm_rish_l2, and MSE; take cluster with
    #      minimum MSE for brain mask; resample and smooth maps first
    #   3. Morphological operations on brain mask; fill holes, opening, LCC, etc; undo resample
    #   4. Look for Globus Pallidus (dark on Rish0 and T2w, =GM on Rish2) and delineate separately;
    #      Article about iron causing decreased T2 in GP compared to Putamen: https://doi.org/10.1007/bf00593967
    #   5. Segment within brain mask using sigm_rish_l0 and sigm_rish_l2 again; 3 classes should
    #      be GM, WM, and CSF+BG
    #   6. Assess CSF+BG mask using T2-weighted image; are there 2 clusters, or significant
    #      low-value voxels? If so, split it into CSF + BG
    # Then the image is 4 voxel classes: BG, CSF, GM, WM

    # (1) recreate rough-mask and apply
    roughmask, roughmask_fn, bimod_mask, bimodmask_fn \
         = rough_bbgmasking(dwi_data, fine_pass=True, shenv=shenv)

    dwi_data *= roughmask[..., None]

    dwi_data_fn = f'{tmpd}/dwi_segbbgdata'
    make_nii(dwi_data, new_aff=affine).to_filename(f'{dwi_data_fn}.nii')


    # (2) segment brain/background within rough-mask
    # calculate RISH images
    shutil.copyfile(str(bvals_path), f'{dwi_data_fn}.bval')
    bvecs_path = str(bvals_path).replace('.bval', '.bvec')
    shutil.copyfile(bvecs_path, f'{dwi_data_fn}.bvec')

    sp.run(f'$DMRIDIR/dmri_rishes {dwi_data_fn} 2',
           shell=True, check=True, env=shenv)

    # run rishes through a sigmoid transform to tamp down the tail of the distribution
    rish_l0_fn = f'{dwi_data_fn}-rish_l0'
    rish0 = check_load_nii(rish_l0_fn).get_fdata()
    rish0_xfm = 1000*sigmoid_xfm(rish0, normp=99) # original normalization was WM-median = 0.75, but this works too

    rish_l2_fn = f'{dwi_data_fn}-rish_l2'
    rish2 = check_load_nii(rish_l2_fn).get_fdata()
    rish2_xfm = 1000*sigmoid_xfm(rish2, normp=99)

    # Fit tensor to b=1000 shell for SSE
    b_dw1 = nzbvals_uniq[0]     # lowest non-zero b-val

    tensor_pp = fit_tensor(dwi_data_fn, roughmask_fn,
                           sse=True, wls=True, save_tensor=False, kurtdir=False,
                           single_b=b_dw1, shenv=shenv)

    sse = check_load_nii(f'{tensor_pp}_sse').get_fdata()

    # Calc MSE
    N = nii.shape[3]
    mse = sse/(N - 6 - 1)

    # mse can also have quite a long tail
    mse_xfm = sigmoid_xfm(mse, normv=1.0)   # keep low MSE values unchanged, forbid values above 1
                                            # !!! recent change to deal with e.g. QNS_0034_01


    # Small amount of gaussian smoothing should help; no need to resample or
    #   import medpy since we're not doing anisotropic smoothing
    rsh0_xfsm = gauss_smooth(rish0_xfm)
    rsh2_xfsm = gauss_smooth(rish2_xfm)
    mse_xfsm  = gauss_smooth(mse_xfm)

    # Re-apply rough-mask to make sure fast ignores the boundary areas
    rsh0_xfsm *= roughmask
    rsh2_xfsm *= roughmask
    mse_xfsm  *= roughmask

    # write out parameter maps for fast to use
    rsh0_sm_fn = f'{rish_l0_fn}_xfsm.nii'
    make_nii(rsh0_xfsm, new_aff=affine).to_filename(rsh0_sm_fn)

    rsh2_sm_fn = f'{rish_l2_fn}_xfsm.nii'
    make_nii(rsh2_xfsm, new_aff=affine).to_filename(rsh2_sm_fn)

    mse_sm_fn = f'{tensor_pp}_mse_xfsm.nii'
    make_nii(mse_xfsm, new_aff=affine).to_filename(mse_sm_fn)


    # Segment into clusters with created parameter maps
    #   [FSL-fast tech note](https://www.fmrib.ox.ac.uk/datasets/techrep/tr04ss2/tr04ss2/node11.html);
    #   the program uses a mixture of gaussians to model the histogram, and applies markov random
    #   fields (see https://ermongroup.github.io/cs228-notes/representation/undirected/) to
    #   regularize the segmentation (i.e. it is smoother in the presence of noise). This process is
    #   iteratively applied, ...
    #   - note, fast has trouble when the boundary background voxels (far outside the brain), in
    #     the roughmask-excluded area have very small, but non-zero values; this can be introduced
    #     by gaussian smoothing
    #   - increasing the spatial smoothness hyperparameter from 0.1 to e.g. 0.9 tends to decrease the
    #     no. of individual dissenting voxels within a larger cluster; drives the segmentation toward
    #     keeping the globus pallidus as well, which is good; even making the parameter 10.0 doesn't
    #     change the segmentation that much, so this will be kept as 1.0

    n_classes = 3
    fast_seg_fn = f'{dwi_data_fn}-seg_fast_n{n_classes}_c3_rsh0_rsh2_mse'

    print(" "*4 + "fine_bbgmasking: calling fast")
    if verbose:
        vstr = '-v'
    else:
        vstr = ''

    sp.run('$FSLDIR/bin/fast'
          f' --class={n_classes} --channels=3 --nobias --Hyper=1.0 {vstr}'
          f' -o {fast_seg_fn} {rsh0_sm_fn} {rsh2_sm_fn} {mse_sm_fn}',
           shell=True, check=True, env=shenv)

    # Seg image should be int type with a small number of values
    pveseg = check_load_nii(f'{fast_seg_fn}_pveseg').get_fdata()

    assert (pveseg.min() == 0) \
       and (pveseg.max() < 255) \
       and np.all(pveseg == pveseg.astype(np.int_)), \
       "pveseg not as expected."

    # measure median values of each tissue type
    rish0_medians = []
    rish2_medians = []
    mse_medians = []
    for i in range(n_classes):
        tf = (pveseg == i+1)

        rish0_medians.append(np.median(rsh0_xfsm[tf]))
        rish2_medians.append(np.median(rsh2_xfsm[tf]))
        mse_medians.append(np.median(mse_xfsm[tf]))      # !!! recent change (mse -> mse_xfsm) when troubleshooting QNS_0034_01

    # Find cluster(s) with minimum MSE (i.e. brain)
    # - in CAN-BIND data, the minimum MSE cluster is usually the brain
    # - in HCP data, usually the 2 minimum MSE clusters are brain
    #                median MSE values are: 0.0013, 0.0063, 0.0319
    #                median rish0_xfsm values are: 508, 293, 23.6
    #                median rish2_xfsm values are: 160, 172, 5.83

    # Perform k-means clustering on median values from the clusters to determine whether
    #   we have 2 brain labels or 1

    X = np.array([rish0_medians, rish2_medians, mse_medians]).T    # X has shape (n_samples, n_features)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Force MSE to matter more with double weighting
    # - after all, the main goal here is to prevent noisy edge voxels in the FA map
    # - this did not always work well on its own; in peripheral areas, MSE is good at
    #   identifying noisy FA maps, but not necessarily in basal areas; then relying too
    #   much on MSE drags low signal areas back into finemask and bimod_mask; so we will
    #   make sure not to expand bimod_mask too much later
    wts = np.array([1,1,2])
    X_wtd = X_std*wts           # !!! this helps with e.g. UBC_0076_01

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X_wtd)

    # Or we could just do k-means on all the data and use a voting process
    #   for now, just treat this as a warning system for odd segmentations
    X2 = np.array([rsh0_xfsm[roughmask], rsh2_xfsm[roughmask], mse_xfsm[roughmask]]).T

    scaler2 = StandardScaler()
    X2_std = scaler2.fit_transform(X2)
    X2_wtd = X2_std*wts

    kmeans2 = KMeans(n_clusters=2)
    kmeans2.fit(X2_wtd)

    # km2_labs_img = np.zeros(roughmask.shape).astype(np.uint8)
    # km2_labs_img[roughmask] = kmeans2.labels_
    # make_nii(km2_labs_img, new_aff=affine).to_filename(f'{tmpd}/fine_bbgmasking-kmlabs.nii')

    # kmeans.cluster_centers_
    #   - ndarray of shape (n_clusters, n_features) = (2, 3)
    #   - Coordinates of cluster centers in standardized feature space
    #   - can get non-standardized values with scaler.inverse_transform(kmeans.cluster_centers_)
    # kmeans.labels_
    #   - vector of length n_samples
    #   - assigns the samples of X a label of 1 or 0 for n_clusters=2

    # Choose the label with maximum rish signals and minimum mse
    cc = kmeans.cluster_centers_
    cc_cmp = (cc[0] > cc[1])
    # if (cc[0,0] > cc[1,0]) and (cc[0,1] > cc[1,1]) and (cc[0,2] < cc[1,2]):
    # elif (cc[0,0] < cc[1,0]) and (cc[0,1] < cc[1,1]) and (cc[0,2] > cc[1,2]):
    if np.all(cc_cmp == [True, True, False]):
        # cluster 0 is brain
        pve_isbrain = (kmeans.labels_ == 0)

    elif np.all(cc_cmp == [False, False, True]):
        # cluster 1 is brain
        pve_isbrain = (kmeans.labels_ == 1)

    else:
        # It seems there is at least 1 ambiguous cluster
        # QNS_0044_02 is one such challenging case -- high MSE in posterior regions within the brain
        print(" "*4 + "fine_bbgmasking: Warning: unable to determine brain cluster from kmeans-medians")

        # Take the cluster with minimum Rish0 to be non-brain background
        i_nonbrain = np.where(rish0_medians == np.min(rish0_medians))[0][0]

        pve_isbrain = np.array([False]*n_classes)

        for i in range(n_classes):
            if i != i_nonbrain:
                pve_isbrain[i] = True

        # # see if kmeans2 gives a clear answer
        # cc2 = kmeans2.cluster_centers_
        # cc2_cmp = (cc2[0] > cc2[1])

        # if np.all(cc2_cmp == [True, True, False]):
        #     # cluster 0 is brain
        #     pve_isbrain2 = (kmeans2.labels_ == 0)

        # elif np.all(cc2_cmp == [False, False, True]):
        #     # cluster 1 is brain
        #     pve_isbrain2 = (kmeans2.labels_ == 1)


    # include any voxel in brain mask that *might* be from the 'brain' clusters (and preserve GP)
    bm_pve = None
    for i in range(n_classes):

        if pve_isbrain[i]:
            new_pve = check_load_nii(f'{fast_seg_fn}_pve_{i}').get_fdata()

            if bm_pve is None:
                bm_pve = new_pve
            else:
                bm_pve += new_pve

        # else:
        #     # check voting from kmeans full clustering
        #     tf = (pveseg[roughmask] == i+1)
        #     lklhood = pve_isbrain2[tf].mean()    # proportion of 'true' voxels that should be classified as brain

        #     if (lklhood > 0.05):
        #         print(" "*4 + f"fine_bbgmasking: Warning: pve with idx={i} not selected, but likelihood is {lklhood:0.3f}")

    finemask = (bm_pve > 0.1)


    # finemask should be essentially contained within bimod_mask, just excluding the
    #   areas with high MSE, but may legitimitely include small regions/protrusions
    #   that got trimmed by the morphops on bimod_mask, e.g. temporal poles.
    #   It may also include non-legitimate areas of background with low signal *and*
    #   low MSE (looking at you, filtered Philips data), so we only include a small
    #   (5 mm) dilation of bimod_mask. !!! This helps with e.g. UBC_0076_01
    voxdims = nii.header.get_zooms()
    vdim_3d = np.mean(voxdims[0:3]).round(3)  # average vox dim in mm

    rad = np.round(5./vdim_3d).astype(int)
    bmmask_dil = ndi.binary_dilation(bimod_mask, structure=ball(rad))

    finemask = finemask & bmmask_dil

    # Morph-ops on fine brain-mask
    # - running finemask through mask_morphops() no longer seems necessary/advantageous;
    #   working within bimod_mask is better, and it's already LCC
    # - but holes in finemask are annoying, especially in the diffusivity maps
    # - finemask stays as bool type, will be written as uint8
    finemask = ndi.binary_fill_holes(finemask)
    assert finemask.dtype == 'bool', 'expected bool dtype from finemask'


    # Touch up bimod_mask -- should include all of finemask
    bimod_mask = bimod_mask | finemask

    if db_seg:
        print("Dropping into ipdb shell to troubleshoot BBG (c to continue anyway)")
        import ipdb; ipdb.set_trace()

    # Trim spots where t2w or dw are practically zero? doesn't seem necessary now
    # dmask1[t2w_res < t2w_res.mean()/300] = False   # T2-W signal < 1 in MCU_0025

    # Consider an inflated mask to clean up the mean-T2w image but keep the CSF/BG transition
    # strelem = ndi.generate_binary_structure(3, 1)
    # strelem = ndi.iterate_structure(strelem, 5)
    # inflmask = finemask.copy()
    # inflmask = ndi.binary_dilation(inflmask, strelem)
    # inflmask = inflmask & roughmask
    # make_nii(inflmask, new_aff=affine).to_filename(f'{tmpd}/mask-inflated.nii')

    mask_perc = 100*finemask.sum()/np.prod(finemask.shape)
    print(" "*4 + f"fine_bbgmasking: fine mask is {mask_perc:0.1f}% of total volume (~ 15–24% is good)")

    # Save masks
    finemask_fn = f'{tmpd}/mask-fine_brainbg.nii'
    make_nii(finemask, new_aff=affine).to_filename(finemask_fn)

    make_nii(bimod_mask, new_aff=affine).to_filename(bimodmask_fn)

    # Create a unified mask from rough (=1), bimod (=2), fine (=3)
    unified_mask = roughmask.astype(np.uint8)
    unified_mask[bimod_mask] = 2
    unified_mask[finemask] = 3

    unified_mask_fn = f'{tmpd}/brainbg_masks.nii'
    make_nii(unified_mask, new_aff=affine).to_filename(unified_mask_fn)


    # Keep roughmask FA image and sse/mse around for QC later
    mv_nii(f'{tensor_pp}_FA',      f'{tmpd}/fine_bbgmasking-dtw_b{b_dw1}_roughmask_FA')
    mv_nii(f'{tensor_pp}_mse_xfsm', f'{tmpd}/fine_bbgmasking-dtw_b{b_dw1}_mse_xfsm')
    mv_nii(f'{tensor_pp}_sse',     f'{tmpd}/fine_bbgmasking-dtw_b{b_dw1}_sse')

    # Clean up
    for fn in glob(f'{tensor_pp}_*.nii*') \
            + glob(f'{dwi_data_fn}_ssb*.*') \
            + glob(f'{dwi_data_fn}-sharm2.nii*') \
            + glob(f'{dwi_data_fn}.*'):
        os.remove(fn)

    masks = (roughmask, bimod_mask, finemask, unified_mask)
    mask_fns = (roughmask_fn, bimodmask_fn, finemask_fn, unified_mask_fn)

    return masks, mask_fns

    # old ideas:
    # # Resample to 2 mm and smooth for consistency, and for anisotropic smoothing
    # rish0_xfm_resamp, _   = resample_image(rish0_xfm, affine)
    # # MSE can't be negative, nor can RISHes; good old splines...
    # rish0_xfm_resamp[rish0_xfm_resamp < 0] = 0
    # # Smooth data in two steps; gaussian and anisotropic
    # rsh0_sm1 = gauss_smooth(rish0_xfm_resamp)
    # rsh0_sm2 = anis_smooth(rsh0_sm1, aff_resamp)
    # rsh0_xfmsm_fn = f'{rish_l0_fn}_xfmsm.nii'
    # make_nii(rsh0_sm2, new_aff=aff_resamp).to_filename(rsh0_xfmsm_fn)
    # # later would need to Reverse resample and write out
    # finemask, _ = resample_image(bm_resamp_morph, affine, fwd=False)
    # assert np.all(finemask.shape == dwi_data.shape[0:3]), "finemask was not resampled properly"


def seg_tissues(dwi_data, bbg_mask, affine=None, shenv=None, db_seg=False):
    """After segmenting brain/background, this function segments the 3 tissue types:
    WM, GM and CSF.

    Pass the bimod brain-mask for bbg_mask

    Will also delineate the globus pallidus separately, since that has such different
    T2-w contrast from the rest of the brain. GP contains much accumulated iron,
    apparently, which causes decreased T2 compared to Putamen: https://doi.org/10.1007/bf00593967

    GP and depression: https://pubmed.ncbi.nlm.nih.gov/9118200/, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4884665/
    """

    if affine is None:
        affine = nii.affine

    # check shell environment
    if shenv is None:
        shenv = os.environ

    assert 'FSLDIR' in shenv, "FSLDIR not set in envirionment"
    assert 'DMRIDIR' in shenv, "DMRIDIR not set in envirionment"

    #   5. Segment within fine brain-mask using sigm_rish_l0 and sigm_rish_l2 again; 3 classes should
    #      be GM, WM, and CSF+BG

    # Calculate parameter maps within mask
    # Total PSD entropy
    print(" "*4 + "seg_tissues: calculating entropy...")
    ent, _ = calc_entropy(dwi_data, bbg_mask, affine=affine)

    # RISH-l0 and RISH-l2 from seg_bbgdata
    rish0 = check_load_nii(f'{tmpd}/dwi_segbbgdata-rish_l0').get_fdata()
    rish0 *= bbg_mask

    rish2 = check_load_nii(f'{tmpd}/dwi_segbbgdata-rish_l2').get_fdata()
    rish2 *= bbg_mask


    # sigmoid transform inputs to tamp down extreme (high) values (mostly around susceptibility artifacts)
    def xfm_input(img, lab):
        img_xfm = 1000*sigmoid_xfm(img, normp=99)
        img_xfm_fn = f'{tmpd}/seg_tissues-{lab}_xfm'
        make_nii(img_xfm, new_aff=affine).to_filename(img_xfm_fn + '.nii')

        return img_xfm, img_xfm_fn

    rish0_xfm, rish0_xfm_fn = xfm_input(rish0, 'rish0')
    rish2_xfm, rish2_xfm_fn = xfm_input(rish2, 'rish2')
    _,         ent_xfm_fn   = xfm_input(ent,   'ent')


    # Three-class tissue segmentation based on (sigmoid-transformed) RISH0, RISH2, and entropy
    #   this works great! N=1
    #   it really works well for GM/CSF/WM, but to try to get GP/STN/RedN as a separate
    #   category, adding in a T2w image does not work well. My best bet is to let GP get
    #   categorized as CSF, then sort it out using T2w signal threshold later.
    n_classes = 3
    fast_seg_fn = f'{tmpd}/seg_tissues-fast_n{n_classes}_c3_ent_rish2_rish0'

    print(" "*4 + "seg_tissues: calling fast")
    if verbose:
        vstr = '-v'
    else:
        vstr = ''

    sp.run(f'$FSLDIR/bin/fast -o {fast_seg_fn}'
           f' --class={n_classes} --channels=3 --nobias {vstr}'
           f' {ent_xfm_fn} {rish2_xfm_fn} {rish0_xfm_fn}',
           shell=True, check=True, env=shenv)


    # Seg image should be int type with a small number of values
    pveseg = check_load_nii(f'{fast_seg_fn}_pveseg').get_fdata()

    assert (pveseg.min() == 0) \
       and (pveseg.max() < 255) \
       and np.all(pveseg == pveseg.astype(np.int_)), \
       "pveseg not as expected."


    # Find cluster with highest RISH2; this is WM
    rish2_medians = []
    for i in range(n_classes):
        mdn = np.median(rish2_xfm[pveseg == i+1])
        rish2_medians.append(mdn)

    i_wm = rish2_medians.index(max(rish2_medians))

    wm_binmask = (pveseg == (i_wm+1))

    wm_pve = check_load_nii(f'{fast_seg_fn}_pve_{i_wm}').get_fdata()

    make_nii(wm_binmask, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-WM_binmask.nii')
    make_nii(wm_pve, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-WM_pve.nii')


    # Cluster with lowest RISH0 is CSF+GP
    rish0_medians = []
    for i in range(n_classes):
        mdn = np.median(rish0_xfm[pveseg == i+1])
        rish0_medians.append(mdn)

    i_csfgp = rish0_medians.index(min(rish0_medians))


    # Then the other cluster is GM
    assert not (i_wm == i_csfgp), "Cluster check problem"

    i_gm = 0
    while (i_gm == i_wm) or (i_gm == i_csfgp):
        i_gm += 1

    gm_binmask = (pveseg == (i_gm+1))

    gm_pve = check_load_nii(f'{fast_seg_fn}_pve_{i_gm}').get_fdata()

    make_nii(gm_binmask, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-GM_binmask.nii')
    make_nii(gm_pve, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-GM_pve.nii')


    # Based on RISH0 + RISH2 + Entropy, Globus Pallidus and STN are mostly classified with CSF,
    #   but can get GM pve > 0 as well.
    # Now differentiate Globus Pallidus from CSF based on T2w intensity

    csfgp_binmask = (pveseg == (i_csfgp+1))

    csfgp_pve = check_load_nii(f'{fast_seg_fn}_pve_{i_csfgp}').get_fdata()

    # T2w has artifact around the susceptibility area too (ears, sinuses)
    t2w = check_load_nii(f'{tmpd}/mean_t2w').get_fdata()
    t2w *= bbg_mask
    # vvv now just using t2w, since we only use the rank of voxels to calculate threshold
    # t2w_xfm = 1000*(sigmoid_xfm(t2w, normp=99)**2)      # square to excentuate contrast
    # t2w_xfm_fn = f'{tmpd}/seg_tissues-t2w_xfm.nii'
    # make_nii(t2w_xfm, new_aff=affine).to_filename(t2w_xfm_fn)

    # Use nii voxel dimensions for img resolution-independence (lookin at you, HCP)
    voxdims = nii.header.get_zooms()
    vdim_xy = np.mean(voxdims[0:2]).round(3)  # x/y plane voxel dimension in mm
    vdim_z  = voxdims[2].round(3)             # z vox dim
    vdim_3d = np.mean(voxdims[0:3]).round(3)  # average vox dim

    # need to consider affine to determine direction of 'front' and 'inferior'
    y_sign = affine[1,1]/abs(affine[1,1])
    z_sign = affine[2,2]/abs(affine[2,2])

    # Erode mask to exclude partial-volume edge voxels
    #   instead, adding ~bbg_mask back in as a component below
    # mask_ero = ndi.binary_erosion(bbg_mask, structure=strelem, iterations=1)

    # Use bbg_mask COM as a spatial reference
    #   then create new mask btw the GP ROIs slightly in front and inferior
    com_ctr = np.array(ndi.center_of_mass(bbg_mask))    # tuple -> array
    com_ctr[1] += y_sign*15./vdim_xy  # forward 15 mm
    com_ctr[2] -= z_sign*15./vdim_z   # down 15 mm

    print(" "*4 + f"seg_tissues: COM co-ords for reference: {tuple(com_ctr.round(1))}")

    # diameter should be 57 voxels * 1.5 mm ~= 86 mm (radius 43 mm)
    #   Directly using 43/vdim_3d gives a radius of ~20–30 voxels depending on image
    #     resolution, which results in a memory error from scipy.ndimage
    #   So use iterations to achieve the same effect
    ball_r = 5
    com_ball = ball(radius=ball_r)

    r_voxunits = 43./vdim_3d
    com_iters = np.ceil(r_voxunits/ball_r).astype(int)

    com_mask = np.zeros(bbg_mask.shape).astype(bool)
    com_mask[tuple(np.round(com_ctr).astype(int))] = 1

    com_mask = ndi.binary_dilation(com_mask, structure=com_ball, iterations=com_iters)

    # as long as the centre is close to the right spot, it's safe to chop off the top and bottom
    #   of the sphere, outside of ~ 45 mm slab around com_ctr
    zlim_h = np.round(com_ctr[2] + 22.5/vdim_z).astype(int)
    zlim_l = np.round(com_ctr[2] - 22.5/vdim_z).astype(int)
    com_mask[:, :, zlim_h:] = 0
    com_mask[:, :, :zlim_l] = 0

    # could also lop off at ~ 27 mm anterior and ~ 35 mm posterior
    ylim_a = np.round(com_ctr[1] + 27./vdim_xy*y_sign).astype(int)
    ylim_p = np.round(com_ctr[1] - 35./vdim_xy*y_sign).astype(int)
    if y_sign < 0:
        com_mask[:, :ylim_a, :] = 0
        com_mask[:, ylim_p:, :] = 0
    else:
        com_mask[:, ylim_a:, :] = 0
        com_mask[:, :ylim_p, :] = 0

    make_nii(com_mask, new_aff=affine).to_filename(f'{tmpd}/seg_tissues-com_mask.nii')

    # com_mask likely includes GP and ventricles, basal cistern, STN, lots of WM and GM, etc
    # Trim down to mask most likely containing GP
    subcort_binmask = (csfgp_pve > 0.25) & com_mask


    # Find threshold for "dark tissue" within the csfgp mask based on T2w signal
    # Using a percentile of WM intensity (within com_mask), since GP and STN are much darker than WM
    # Now using 33rd percentile instead of median: even stricter, fewer false-positives
    #   -> this results in _much_ fewer labeled features below (fewer false-positives), compared to
    #      thresholds using e.g. kmeans, otsu, or minimum algorithm
    #   -> old tries: t2w_vox = t2w_xfm[subcort_binmask]
    #                 thresh = threshold_otsu(t2w_vox)
    #                 thresh_min = threshold_minimum(t2w_vox)
    #   -> also tried KMeans(n_clusters=2)
    #   -> ^^ These thresholds didn't really work: estimated thresh=360 or more when the GP and STN are << 100
    thresh_wm = np.percentile(t2w[(wm_pve > 0.9) & com_mask], 33)

    print(" "*4 + f"seg_tissues: using T2w thresh {thresh_wm:0.2f}")

    subcort_binmask = subcort_binmask & (t2w < thresh_wm)


    # Areas touching the bbg-mask boundary are likely areas of signal dropout due to susceptibility
    #   artifact, but not because of iron as in GP; so these will be excluded
    subcort_susc_binmask = subcort_binmask | ~bbg_mask     # this works b/c bbg_mask is array of np.bool_ type

    # Then the largest LCC is partial volume/susceptibility artifact/background
    #   and the next 2 LCCs are likely to be/include GP
    # label image for largest connected component with square connectivity
    labs_img, n_feat = ndi.label(subcort_susc_binmask)                             # labs_img has dtype int32

    lcc_lab = stats.mode(labs_img[labs_img > 0], axis=None).mode[0]       # mode of non-zero labels

    lab_sizes = ndi.sum(subcort_susc_binmask, labs_img, index=range(1, n_feat+1))  # 1D array with len n_feat
    idx_sorted = np.argsort(-1*lab_sizes)                                 # indices in descending order or size

    if verbose:
        print(" "*4 + f"seg_tissues: initial cluster sizes: {lab_sizes[idx_sorted[1:9]]}")
        # make_nii(labs_img, new_aff=affine).to_filename(f'{tmpd}/seg_tissues-labs_img_init.nii')

    # Drop first LCC from label mask
    #   however there is a danger that the susceptibility cluster will be very big, and touching GP
    #   etc through a snaking bridge; therefore limit the effect of this step to ~ 6.5 mm or so
    if verbose:
        print(" "*4 + f"seg_tissues: ignoring first cluster with size {lab_sizes[idx_sorted[0]]:.2g}")

    rad = np.round(6.5/vdim_3d).astype(int)
    bbg_mask_ero = ndi.binary_erosion(bbg_mask, structure=ball(rad))
    excl_mask = (labs_img == lcc_lab) & ~bbg_mask_ero

    labs_mask = (labs_img > 0) & ~excl_mask

    # labs_mask can be a bit sloppy/rough
    #  - consider a binary closing operation before filling holes
    #  - even a dilation, but avoid areas that are 100% for sure WM, or above threshold
    #  -> ^^^ this doesn't seem necessary now that I'm using csfgp_pve > 0.25 above, rather
    #         than csfgp_binmask; labs_mask is much more complete, so just fill holes
    #  - note in subjects like MGH_1016, labs_mask can be very permissive

    labs_mask = ndi.binary_fill_holes(labs_mask)  # default square connectivity is more filly
                                                  # (diagonal gaps are still considered holes to fill)

    # cluster mass may be calculated from cluster size and *darkness*
    clust_mass_img = -1*(t2w - thresh_wm)*labs_mask

    # STN, being fairly flat/thin in the lateral direction, is mostly demolished by an erosion operation,
    #   even in high-res HCP data; but we still need a strategy to split the GP and STN cluster, which can
    #   be connected by a filament; the scheme is:
    #   - erode labs_mask
    #   - label GP clusters
    #   - dilate GP clusters, take union with uneroded mask as GP clusters
    #   - subtract GP clusters from uneroded mask
    #   - label remaining uneroded mask
    #   - find STN and RedN in new labels
    labs_mask_ero = ndi.binary_erosion(labs_mask)  # using square connectivity

    if verbose:
        print(" "*4 + f"seg_tissues: before erosion, labs_mask has {labs_mask.sum()} voxels with total mass: {clust_mass_img.sum():.2g}")
        # make_nii(clust_mass_img, new_aff=affine).to_filename(f'{tmpd}/seg_tissues-clust_mass_img.nii')
        print(" "*4 + f"seg_tissues: after erosion, labs_mask_ero has {labs_mask_ero.sum()} voxels with total mass: {(clust_mass_img*labs_mask_ero).sum():.2g}")
        # make_nii(labs_img, new_aff=affine).to_filename(f'{tmpd}/seg_tissues-labs_img_ero.nii')

    labs_img, n_feat = ndi.label(labs_mask_ero)    # GP from eroded labs

    lab_sizes = ndi.sum(labs_mask_ero, labs_img, index=range(1, n_feat+1))  # 1D array with len n_feat
    idx_sorted = np.argsort(-1*lab_sizes)                                   # indices in descending order of size


    # Calculate/report cluster sizes and centres of mass

    # e.g. first 14 sizes clusters and locations
    #
    # count    coords
    # 1.5 million          # background and susceptibility
    #  134   (54, 48, 29)  # GP (L)
    #  123   (41, 49, 29)  # GP (R)
    #   91   (45, 27, 26)  # boundary of cerebellum
    #   34   (51, 32, 13)  # cerebellum
    #   29   (44, 32, 53)  # superior parietal
    #   28   (50, 61, 26)  # inferior frontal
    #   26   (40, 32, 13)  # cerebellum
    #   25   (54, 72, 39)  # frontal pole
    #   22   (42, 27, 52)  # superior parietal
    #   19   (43, 45, 25)  # sub-thalamic nucleus (R)
    #   19   (43, 45, 25)  # ""
    #   18   (50, 44, 24)  # sub-thalamic nucleus (L)
    #   12   (52, 34, 23)  # boundary of cerebellum

    class Cluster:
        # Small masks defining possible locations of GP, STN, RedN

        # Initializer
        def __init__(self, mask):
            self.mask = mask

            self.anat = None

            self.update()

        def update(self, mask=None):
            if mask is None:
                mask = self.mask

            # calculate attributes based on mask
            self.size = mask.sum()
            self.com = ndi.center_of_mass(clust_mass_img*mask)  # tuple of img/voxel coords, e.g. (54, 49, 29)
            self.xdist = vdim_xy*(self.com[0] - com_ctr[0])     # distance of COM from midline in mm
            self.side = abs(self.xdist)/self.xdist              # +/- 1


    def filter_clusters(n_max):
        """relies on the currently set labs_img, n_feat, idx_sorted"""

        n_clust = min(n_feat, n_max)
        clusters = []

        if n_clust > 0:
            print(f"; largest {n_clust}:")
            print(" "*17 + f"idx  size  COM")

        rej_msgs = []
        for idx in range(0, n_clust):
            mask = (labs_img == (idx_sorted[idx] + 1))
            clust = Cluster(mask)

            if (abs(clust.xdist) < 3):
                # cluster right on the midline (display COM rounded to nearest 0.5)
                rej_msgs.append(" "*17 + f"  - {clust.size:>5d}  {tuple(np.round(clust.com, 1))}")
                continue

            clusters.append(clust)

            print(" "*17 + f"{idx-len(rej_msgs):>3d} {clust.size:>5d}  {tuple(np.round(clust.com, 1))}")

        n_clust = len(clusters)

        if len(rej_msgs) > 0:
            print("\n" + " "*17 + f"rejected the following midline clusters, leaving {n_clust}:")
            for msg in rej_msgs:
                print(msg)

        return clusters, n_clust


    def find_clust_pair(c1_idx, avoid_idx=None, tight=False):
        """Try to find a matching pair for the given cluster;
           - it should be roughly the same size, on the other side, close in y, z, and abs(x)
           - relies on already defined variables n_clust, clusters
        """

        if tight:
            # limits are tighter for e.g. red nucleus
            xdlim = 2.5
            yzlim = 2.5
        else:
            xdlim = 2.0*vdim_xy    # 5 mm for CBN01 data, 3 mm for HCP (originally 3.75 absolute)
            yzlim = 3.2*vdim_3d    # 6.25 mm for CBN01 data, 3.75 mm for HCP (originally 4.25 absolute)
                                   #   e.g. CAM_0011_02 is so angled that yzdist=2.9*vdim_3d b/c y-coords are off

        c1 = clusters[c1_idx]
        pair_found = False

        c2_idx = None              # in case c1_idx+1 == n_clust

        for c2_idx in range(c1_idx+1, n_clust):

            if c2_idx == avoid_idx:
                continue

            c2 = clusters[c2_idx]

            if (abs(c1.size - c2.size)/c1.size > 0.80):
                # clusters getting too different in size (later clusters will be smaller)
                if verbose:
                    print(" "*4 + f"seg_tissues:   rejecting cluster {c2_idx} and following (size)")
                break

            pair_yzdist = np.sqrt((vdim_xy*(c1.com[1] - c2.com[1]))**2 + \
                                    (vdim_z*(c1.com[2] - c2.com[2]))**2)

            if (c1.side*c2.side > 0):
                # clusters on same side
                if verbose:
                    print(" "*4 + f"seg_tissues:   rejecting cluster {c2_idx} (same side)")

            elif (abs(c1.xdist + c2.xdist) > xdlim):
                # inconsistent laterality
                if verbose:
                    print(" "*4 + f"seg_tissues:   rejecting cluster {c2_idx} (inconsistent laterality)")

            elif (pair_yzdist > yzlim):
                # inconsistent SI and AP
                if verbose:
                    print(" "*4 + f"seg_tissues:   rejecting cluster {c2_idx} (inconsistent SI/AP)")

            else:
                pair_found = True
                break

        return pair_found, c2_idx


    # Identify a GP cluster to start
    #   - usually they're much bigger than any other clusters
    #   - GP clusters should be close to saggittal midline, but not right on it
    #   - using the COM of bbg_mask as a ref
    #   - this examines clusters in descending order of size

    print(" "*4 + f"seg_tissues: found {n_feat} (eroded) clusters as candidates for GP", end="")
    clusters, n_clust = filter_clusters(8)     # GP ought to be the first 2, really

    gp1_likely = False

    for gp1_idx in range(0, n_clust-1):

        c1 = clusters[gp1_idx]

        if (abs(c1.xdist) > 12) and (abs(c1.xdist) < 24):
            # GP distance from midline should be ~ 18 mm
            gp1_likely = True

        if gp1_likely:
            # Try to find a matching pair for the first GP cluster
            #   it should be roughly the same size, on the other side, close in y, z, and abs(x)
            print(" "*4 + f"seg_tissues: searching for match to GP cluster {gp1_idx}")

            gp_pair, gp2_idx = find_clust_pair(gp1_idx)

            if gp_pair:
                c2 = clusters[gp2_idx]
                break

            else:
                # need to try the next cluster for c1
                gp1_likely = False

    if gp1_likely and gp_pair:
        # define the bilateral mask
        print(" "*4 + f"seg_tissues: found GP pair ({gp1_idx} + {gp2_idx})")
        gp_bilatmask = c1.mask | c2.mask

        # add a dilation to counter the earlier erosion
        strelem = ndi.generate_binary_structure(3, 2)       # first order diagonals, but not 3D corners
        gp_bilatmask = ndi.binary_dilation(gp_bilatmask, structure=strelem)

        # make sure we're not introducing completely new voxels compared to original labs
        # assert np.all(labs_mask[gp_bilatmask] == True), "expected gp_bilatmask to be totally covered by labs_mask"
        # ^^^ with non-square connectivity for the dilation, this assertion fails (for 358 voxels!)
        gp_bilatmask = gp_bilatmask & labs_mask

        print(" "*4 + f"seg_tissues: bilateral GP mask volume: {gp_bilatmask.sum()} voxels = {gp_bilatmask.sum()*vdim_3d**3:.1f} mm^3")

        gp_xdmag = 0.5*(abs(c1.xdist) + abs(c2.xdist))  # average magnitude of GP x-dist, in mm
        gp_yloc = 0.5*vdim_xy*(c1.com[1] + c2.com[1])   # average y location, mm
        gp_zloc = 0.5*vdim_z*(c1.com[2] + c2.com[2])    # average z location, mm

        print(" "*4 + f"seg_tissues: GP COM locs for reference: xdmag={gp_xdmag/vdim_xy:.1f}, yloc={gp_yloc/vdim_xy:.1f}, zloc={gp_zloc/vdim_z:.1f}")

        # Prepare to identify other important subcortical clusters
        # Relabel original labs_mask with GP clusters removed
        labs_mask = labs_mask ^ gp_bilatmask     # xor logic: "A or B but not (both or neither)"

        labs_img, n_feat = ndi.label(labs_mask)  # new labels from uneroded mask without GP

        lab_sizes = ndi.sum(labs_mask, labs_img, index=range(1, n_feat+1))  # 1D array with len n_feat
        idx_sorted = np.argsort(-1*lab_sizes)                               # indices in descending order of size

        make_nii(labs_img, new_aff=affine).to_filename(f'{tmpd}/seg_tissues-labs_img_no_GP.nii')

        # Now examine the new labels -- should be able to pull out STN and RedN from HCP data, at least
        print(" "*4 + f"seg_tissues: found {n_feat-1} clusters as candidates for STN", end="")
        clusters, n_clust = filter_clusters(16)  # they should be early, but some will be midline

        # 2 of the next biggest clusters are likely to be subthalamic nuclei (STN), which have high iron:
        #   - Iron concentration linked to structural connectivity in the subthalamic nucleus:
        #     implications for deep brain stimulation; https://doi.org/10.3171/2018.8.jns18531
        #   - Direct visualization of the subthalamic nucleus and its iron distribution using
        #     high-resolution susceptibility mapping; https://doi.org/10.1002/hbm.21404

        # STN are smaller flat, curved structures closer to midline (~ 9 mm), posterior and inferior to GP
        #  - clusters can show up posterior, *superior*, and at the same laterality as GP, which is posterior thalamus
        #  - another pair of clusters is more medial and round, within the curve of STN; this is red nucleus,
        #    with COM about 5 mm from midline

        # Check the size of the first cluster
        #  in future, maybe go back and rethreshold csfgp if necessary, to split up STN and RedN clusters,
        #  as is needed in MGH_1016; or experiment with adding an erosion step
        #  e.g. if first cluster is more than 1200 mm^3, we may have a problem with clusters too big
        clust0_sz = clusters[0].mask.sum()*vdim_3d**3
        if clust0_sz > 1200:
            print(" "*4 + f"seg_tissues: note, first cluster is large: {clust0_sz:.0f} mm^3")


        # Identify STN cluster
        #   - COM should also be inferior and posterior to GP
        print(" "*4 + f"seg_tissues: searching top {n_clust} candidates for STN clusters")
        stn1_likely = False
        stn1_orig_idx = None

        for stn1_idx in range(0, n_clust):

            c1 = clusters[stn1_idx]

            stn_gp_ydiff = vdim_xy*c1.com[1] - gp_yloc    # positive for y_sign = -1
            stn_gp_zdiff = vdim_z*c1.com[2] - gp_zloc     # negative for z_sign = +1

            if (stn_gp_ydiff*y_sign > 0) or (abs(stn_gp_ydiff) > 22):
                # cluster anterior to GP, or too posterior to GP (should be ~ 15 mm)
                if verbose:
                    print(" "*4 + f"seg_tissues:   rejecting cluster {stn1_idx} (ydiff={stn_gp_ydiff*y_sign:.1f})")

            elif (stn_gp_zdiff*z_sign > 2) or (abs(stn_gp_zdiff) > 14):
                # cluster superior to GP (with 2 mm buffer), or too inferior to GP (should be ~ 6 mm, but I have seen 10.6 mm)
                if verbose:
                    print(" "*4 + f"seg_tissues:   rejecting cluster {stn1_idx} (zdiff={stn_gp_zdiff*z_sign:.1f})")

            elif (abs(c1.xdist) > (gp_xdmag-2)):
                # cluster closer to midline than GP
                if verbose:
                    print(" "*4 + f"seg_tissues:   rejecting cluster {stn1_idx} (xdist={c1.xdist:.1f})")

            else:
                stn1_likely = True


            if stn1_likely:
                # find a matching pair for the stn1 cluster
                print(" "*4 + f"seg_tissues: searching for match to STN cluster {stn1_idx}")

                stn_pair, stn2_idx = find_clust_pair(stn1_idx)

                if stn_pair:
                    c2 = clusters[stn2_idx]
                    break

                else:
                    # need to try the next cluster for c1
                    print(" "*4 + f"seg_tissues: no pair, moving to next STN1 cluster")
                    stn1_likely = False

                    if stn1_orig_idx is None:
                        stn1_orig_idx = stn1_idx

        if stn1_likely and stn_pair:

            stn_xdmag = 0.5*(abs(c1.xdist) + abs(c2.xdist))
            stn_yloc = 0.5*vdim_xy*(c1.com[1] + c2.com[1])
            stn_zloc = 0.5*vdim_z*(c1.com[2] + c2.com[2])

            # check for combined STN + RedN cluster
            def check_clust_com(clust, label):
                '''determine whether cluster COM is actually a cluster voxel, or in no-mans land'''
                if not clust.mask[tuple(np.round(clust.com).astype(int))]:
                    print(" "*4 + f"seg_tissues: cluster {label} may actually be merge of multiple clusters")

            check_clust_com(c1, 'STN1')
            check_clust_com(c2, 'STN2')

            # Check whether these clusters are STN or more likely red nucleus and report
            # closer to 9 or 5 mm ?
            if (abs(stn_xdmag - 5) < abs(stn_xdmag - 9)):
                # more likely red nucleus
                print(" "*4 + f"seg_tissues: new cluster pair ({stn1_idx} + {stn2_idx}) likely to be red nucleus (xdist={stn_xdmag:.1f})")

                redn_bilatmask = c1.mask | c2.mask
                stn_bilatmask = np.zeros(bbg_mask.shape).astype(bool)

            else:
                # does seem to be STN (can be combo)
                print(" "*4 + f"seg_tissues: found STN pair ({stn1_idx} + {stn2_idx})")

                stn_bilatmask = c1.mask | c2.mask

                print(" "*4 + f"seg_tissues: STN COM locs for reference: xdmag={stn_xdmag/vdim_xy:.1f}, yloc={stn_yloc/vdim_xy:.1f}, zloc={stn_zloc/vdim_z:.1f}")

                # Finally, identify Red nucleus pair
                # - should be very close to STN in y and z (~ 3 mm posterior and 4 mm superior)
                # - closer to the midline (~ 5 mm)

                print(" "*4 + f"seg_tissues: searching rest of the {n_clust-stn1_idx-2} clusters for RedN")
                rdn1_likely = False
                rdn1_orig_idx = None

                for rdn1_idx in range(stn1_idx+1, n_clust):

                    if rdn1_idx == stn2_idx:
                        continue

                    c1 = clusters[rdn1_idx]

                    rdn_stn_ydiff = vdim_xy*c1.com[1] - stn_yloc   # positive for y_sign = -1
                    rdn_stn_zdiff = vdim_z*c1.com[2] - stn_zloc    # positive for z_sign = +1

                    if (rdn_stn_ydiff*y_sign > 0) or (abs(rdn_stn_ydiff) > 8):
                        # cluster anterior to STN, or too posterior to STN (should be ~ 3 mm)
                        if verbose:
                            print(" "*4 + f"seg_tissues:   rejecting cluster {rdn1_idx} (ydiff={rdn_stn_ydiff*y_sign:.1f})")

                    elif (rdn_stn_zdiff*z_sign < -2) or (abs(rdn_stn_zdiff) > 10):
                        # cluster inferior to STN (with 2 mm buffer), or too superior to STN (should be ~ 5 mm)
                        if verbose:
                            print(" "*4 + f"seg_tissues:   rejecting cluster {rdn1_idx} (zdiff={rdn_stn_zdiff*z_sign:.1f})")

                    elif (abs(c1.xdist) > stn_xdmag):
                        # cluster farther from midline than STN
                        if verbose:
                            print(" "*4 + f"seg_tissues:   rejecting cluster {rdn1_idx} (xdist={c1.xdist:.1f})")

                    else:
                        rdn1_likely = True


                    if rdn1_likely:
                        # find a matching pair for the rdn1 cluster
                        print(" "*4 + f"seg_tissues: searching for match to RedN cluster {rdn1_idx}")

                        rdn_pair, rdn2_idx = find_clust_pair(rdn1_idx, avoid_idx=stn2_idx, tight=True)

                        if rdn_pair:
                            c2 = clusters[rdn2_idx]
                            break

                        else:
                            # need to try the next cluster for c1
                            print(" "*4 + f"seg_tissues: no pair, moving to next RedN1 cluster")
                            rdn1_likely = False

                            if rdn1_orig_idx is None:
                                rdn1_orig_idx = rdn1_idx

                if rdn1_likely and rdn_pair:
                    print(" "*4 + f"seg_tissues: found RedN pair ({rdn1_idx} + {rdn2_idx})")
                    redn_bilatmask = c1.mask | c2.mask

                else:
                    print(" "*4 + f"seg_tissues: did not find another pair", end="")

                    if rdn1_orig_idx is not None:
                        print(f"; orig guess for RedN was cluster {rdn1_orig_idx}; using unilateral mask")
                        # could put a check here on whether contralateral COM of clusters[rdn1_orig_idx]
                        # is in one of the STN masks
                        redn_bilatmask = clusters[rdn1_orig_idx].mask
                    else:
                        print("")
                        redn_bilatmask = np.zeros(bbg_mask.shape).astype(bool)

        else:
            print(" "*4 + f"seg_tissues: did not find another pair", end="")

            if stn1_orig_idx is not None:
                print(f"; orig guess for STN was cluster {stn1_orig_idx}; using unilateral mask")
                # could put a check here on whether contralateral COM of clusters[stn1_orig_idx]
                # is in one of the GP masks
                stn_bilatmask = clusters[stn1_orig_idx].mask
            else:
                print("; STN may be included in GP bilatmask")
                stn_bilatmask = np.zeros(bbg_mask.shape).astype(bool)

            redn_bilatmask = np.zeros(bbg_mask.shape).astype(bool)

        # More areas could be targeted: posterior thalamus, cerebellum, etc.; but if the above works, we're
        #   doing pretty well

    else:
        print(" "*4 + "seg_tissues: failed to find a cluster pair likely to be GP")

        db_seg = True

        gp_bilatmask = np.zeros(bbg_mask.shape).astype(bool)
        stn_bilatmask = gp_bilatmask.copy()
        redn_bilatmask = gp_bilatmask.copy()

    if db_seg:
        print("Dropping into ipdb shell to troubleshoot seg (c to continue anyway)")
        import ipdb; ipdb.set_trace()

    if np.any(gp_bilatmask):
        make_nii(gp_bilatmask, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-GP_bilatmask.nii')

    if np.any(stn_bilatmask):
        print(" "*4 + f"seg_tissues: total STN mask volume: {stn_bilatmask.sum()} voxels = {stn_bilatmask.sum()*vdim_3d**3:.1f} mm^3")
        make_nii(stn_bilatmask, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-STN_bilatmask.nii')

    if np.any(redn_bilatmask):
        print(" "*4 + f"seg_tissues: total RedN mask volume: {redn_bilatmask.sum()} voxels = {redn_bilatmask.sum()*vdim_3d**3:.1f} mm^3")
        make_nii(redn_bilatmask, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-RedN_bilatmask.nii')

    not_cluster_mask = ~(gp_bilatmask | stn_bilatmask | redn_bilatmask)

    # We might have stolen a bit from GM and WM by using a liberal PVE-based CSF mask
    gm_binmask = gm_binmask & not_cluster_mask
    wm_binmask = wm_binmask & not_cluster_mask

    # What's left over with signal << CSF will be categorized as 'other'
    #   probably mostly susceptibility artifact / partial volume / low signal areas
    other_binmask = csfgp_binmask & (t2w < thresh_wm) & not_cluster_mask

    if np.any(other_binmask):
        make_nii(other_binmask, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-other_binmask.nii')

    # And the rest will be categorized as CSF
    csf_binmask = csfgp_binmask & ~other_binmask & not_cluster_mask

    csf_pve = np.zeros(bbg_mask.shape)
    csf_pve[csf_binmask] = csfgp_pve[csf_binmask]

    make_nii(csf_binmask, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-CSF_binmask.nii')
    make_nii(csf_pve, new_aff=affine).to_filename(f'{tmpd}/tissue_seg-CSF_pve.nii')


    # Return masks and pve images
    binary_masks = (wm_binmask, gm_binmask, csf_binmask, gp_bilatmask, stn_bilatmask, redn_bilatmask, other_binmask)
    pve_imgs = (wm_pve, gm_pve, csf_pve)

    return binary_masks, pve_imgs


# vvv not using this any more, but there was nothing wrong with the code
# import medpy.filter as mpf
# import warnings  # for ^^^
#
# def anis_smooth(img, affine):
#     """Anisotropic diffusion filter from medpy library

#     returns smoothed ndimage
#     """

#     # medpy function expects float valued img (0 -> 1)
#     a = np.min(img)
#     b = np.max(img)
#     flv_data = (img - a)/(b - a)

#     # medpy raises a future warning because of chained assignment in smoothing.py
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=FutureWarning)

#         # anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1)
#         sm_img = mpf.anisotropic_diffusion(flv_data, niter=3,
#                                            voxelspacing=nib.affines.voxel_sizes(affine))

#     # get voxel values back
#     sm_img = sm_img*(b - a) + a

#     return sm_img


# vvv not using this any more, but there was nothing wrong with the code
# def resample_image(img, affine, scale=2., fwd=True):
#     """Resample 3D or 4D array to 2 mm (or specified) spacing; returns zoomed img and adjusted affine.

#     If fwd = False, does reverse resampling operation.

#     Recommend to create a _new_ nifti using the rescaled affine and the orig header.
#     """

#     if (img.dtype == 'bool') or (img.dtype == 'uint8'):
#         # nearest-neighbour interp for integers
#         input_isints = True
#         splorder = 0
#     else:
#         # cubic spline for floats
#         input_isints = False
#         splorder = 3

#     # Get current voxel scales and calculate resampling factor
#     #   uses nibabel dev strategy (https://github.com/nipy/nibabel/issues/670)
#     RZS = affine[:3, :3]                            # rotation, zoom, skew portion
#     orig_scales = np.sqrt(np.sum(RZS**2, axis=0))   # should give the same values as nib.affines.voxel_sizes(hdr.get_best_affine()) and hdr.get_zooms(); gets scale factors from magnitudes of col vectors in affine (see https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati)
#     trgt_scales = np.array([scale]*3)

#     img_zoom = orig_scales/trgt_scales

#     if not fwd:
#         img_zoom = 1./img_zoom


#     # Handle 3D/4D imgs by making all 4D
#     if img.ndim == 3:
#         input_is4d = False
#         img = img.reshape(np.append(img.shape, 1))
#     else:
#         input_is4d = True

#     vol_indices = range(img.shape[3])

#     # Pre-allocate array
#     new_shape = (img.shape*np.append(img_zoom, 1.)).round().astype(np.int_)
#     new_img = np.zeros(new_shape)

#     # Resample volume-at-a-time
#     if input_is4d:
#         print("Resampling vols: ", end='')

#     for v in vol_indices:
#         if input_is4d:
#             print(v, end='')

#         # Rescale img using the resampling factor
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=UserWarning)

#             new_img[..., v] = ndi.zoom(img[...,v], img_zoom,
#                                         order=splorder, mode='nearest')     # mode: how to handle edge

#     if input_is4d:
#         print('')

#     # Return 3D array if input was 3D
#     if new_img.shape[3] == 1:
#         new_img = new_img.squeeze(axis=3)


#     # Make sure to return integers for masks and labels
#     if input_isints:
#         # assert np.all(new_img == new_img.astype(img.dtype))
#         new_img = new_img.astype(img.dtype)


#     # Rescale affine to match
#     aff_zoom = 1./img_zoom

#     new_aff = np.array(affine, copy=True)

#     if fwd:
#         new_aff[:3, :3] = RZS*np.diag(aff_zoom)

#     else:
#         #new_aff[:3, :3] = RZS*np.diag(img_zoom)*np.diag(aff_zoom)
#         pass


#     # Take care of the translation
#     #   this is necessary since the (0,0,0) co-ord in voxel space is the _center_ of
#     #   the first voxel (see http://nipy.org/nibabel/coordinate_systems.html)
#     T = affine[:3, 3]   # x,y,z translations as vector
#     T_sign = np.sign(np.diagonal(affine[:3, :3]))

#     if fwd:
#         # negative values for going to higher resolution
#         T_offset = 0.5*orig_scales*(aff_zoom - 1)   # x,y,z correction based on my diagram
#         new_aff[:3, 3] = T + T_offset*T_sign

#     else:
#         # T_offset = 0.5*orig_scales*(img_zoom - 1)
#         # new_aff[:3, 3] = T - T_offset*T_sign + T_offset*T_sign  # pointless, right? Yet...
#         pass

#     return new_img, new_aff

#     # Old ideas:
#     #  - less reliable way from medpy didn't properly handle sform/qform; didn't work in fsleyes
#     #    data_res, hdr_res = mpf.resample(img, hdr, trgt_scales)
#     #  - more hassle and hacky way that cleans up set_zooms' deficiencies
#     #    header.set_zooms([x,y,z]), sets the pixdim; can just use this with ndi.zoom?
#     #    header.set_qform(affine, code), may be necessary otherwise
#     #    hdr.set_zooms(trgt_scales)  # changes qform only, regardless of codes!
#                                      # see http://nipy.org/nibabel/nifti_images.html for codes
#     #    hdr.set_sform(hdr.get_qform(), code=1)
