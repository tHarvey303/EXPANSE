# Gratefully modified from aperpy (https://github.com/astrowhit/aperpy/tree/main), all credit to Weaver et al for the original code.

from astropy.io import fits
import numpy as np
import os, sys
from astropy.nddata import block_reduce
from photutils.psf import create_matching_kernel, SplitCosineBellWindow
from scipy.ndimage import zoom
from astropy.visualization import simple_norm
from photutils.centroids import centroid_2dg
from astropy.convolution import convolve_fft
import time
from astropy.table import Table
import scipy
from copy import copy
from astropy.stats import mad_std
from photutils.centroids import centroid_com
import cmasher as cmr
import astropy.units as u
import time
import matplotlib.pyplot as plt
from photutils.detection import find_peaks
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from astropy.modeling.models import Linear1D
import astropy.units as u
import os
import glob
from math import atan2,degrees
from scipy.stats import loglaplace, chi2
from astropy.visualization import ImageNormalize
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import cv2
from astropy.wcs import WCS
from scipy.ndimage import shift
from photutils import CircularAperture, aperture_photometry
from astropy.table import hstack
import pickle
from astropy.stats import sigma_clip
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)
np.errstate(invalid='ignore')
from astropy.visualization import (MinMaxInterval, LinearStretch, SqrtStretch, AsinhStretch,
                                   ImageNormalize)
from astropy.visualization import lupton_rgb


def make_psf(filters, img_paths, outdir, kernel_dir, img_pixel_scale = 0.03, psf_fov = 4, phot_zp = 28.08, match_band = 'F444W', skyext = '', oversample = 3, alpha = 0.3, beta = 0.15, pypher_r = 3e-3, maglim = (18.0, 24.0), method = 'pypher'):
    '''
    Generate PSF (Point Spread Function) for a given set of images.

    Parameters:
    img_paths (dict): A dictionary containing the paths to the images for each filter.
    img_pixel_scale (float): The pixel scale of the images in arcseconds per pixel.
    psf_fov (float): The field of view (FOV) of the PSF in arcseconds.
    filters (list): A list of filters for which the PSF needs to be generated.
    phot_zp (float or dict of bands): The photometric zero point for the images.
    match_band (str): The reference band for matching the PSFs.
    skyext (str): The extension of the sky background in the FITS file.
    outdir (str): The directory where the PSF files will be saved.
    oversample (int): The oversampling factor for the PSF. The output PSF will be sampled at
        the original pixel scale but this factor is used to derive the kernel.
    alpha (float): The alpha parameter for the PSF generation. - not used if method is 'pypher'.
    beta (float): The beta parameter for the PSF generation. - not used if method is 'pypher'.
    pypher_r (float): The radius parameter for the PyPHER algorithm. - only used if method is 'pypher'.
    maglim (float): The magnitude limit for the PSF generation.
    method (str, optional): The method to be used for PSF generation. Defaults to 'pypher'.

    Returns:
    None
    '''
    #outdir = outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    plotdir = os.path.join(outdir,'diagnostics/')
    print(plotdir)
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)

    target_filter = match_band
    print('target filter',target_filter)
    
    image_path = img_paths[target_filter]

    if type(image_path) != list:
        image_path = [image_path]
    
    #for i, path in enumerate(image_path):
        

    hdr = fits.getheader(image_path[0])
    
    use_filters = [match_band] + [f for f in filters if f != match_band]
    if type(phot_zp) == float:
        phot_zp = {f:phot_zp for f in filters}
        

    for pfilt in use_filters:
    
        print(f'Finding stars for {pfilt}...')
        filenames = img_paths[pfilt]
        if type(filenames) != list:
            filenames = [filenames]
        suffix = '.fits' + filenames[0].split('.fits')[-1]
        # Get path before the filename
        path = os.path.dirname(filenames[0])
        starname = filenames[0].replace(suffix, '_star_cat.fits').replace(path, outdir)
        outname = os.path.join(outdir, f'{pfilt}.fits')

        if len(glob.glob(outdir+'*'+pfilt+'*'+'psf.fits')) > 0:
            print(f'PSFs already exist for {pfilt} -- skipping!')
            if pfilt == target_filter:
                target_psf = fits.getdata(glob.glob(outdir+'*'+target_filter+'*'+'psf.fits')[0])
            continue

        #print(filename)
        print(starname)
        # Run in loop


        snr_lim = 1000
        sigma = 2.8 #if pfilt in ['f090w'] else 4.0
        showme=True

        peaks_all, stars_all = [], []
        ra_all, dec_all, ids_all = [], [], []
        for i, filename in enumerate(filenames):
            peaks, stars = find_stars(filename, outdir=outdir, plotdir=plotdir, label=pfilt, zp=phot_zp[pfilt])
            peaks_all.append(peaks)
            stars_all.append(stars)
            print(f'Found {len(peaks)} bright sources in {filename}')
            

            ok = (peaks['mag'] > maglim[0]) & ( peaks['mag'] < maglim[1] )
            ra, dec, ids = peaks['ra'][ok], peaks['dec'][ok], peaks['id'][ok]
            ra_all.append(ra)
            dec_all.append(dec)
            ids_all.append(ids)


        print(f'Found {np.sum([len(i) for i in peaks_all])} sources with {maglim[0]} < mag < {maglim[1]} in {pfilt}...')
        print(f'Processing PSF...')
        # Define the PSF object
        pixsize = int(psf_fov/img_pixel_scale)
        print(f'PSF model dimensions: {pixsize} pix = {pixsize*img_pixel_scale} arcsec ({psf_fov} arcsec requested)')
        psf = PSF(images=filenames, all_x=ra_all, all_y=dec_all, all_ids=ids_all, pixsize=pixsize, pixelscale=img_pixel_scale)#101)
        # Center PSF model and measure
        psf.center()
        psf.measure()
        psf.select(snr_lim=snr_lim, dshift_lim=3, mask_lim=0.99, showme=showme, nsig=30)
        print(f'First selection {np.sum(psf.ok)} stars left')
        psf.stack(sigma=sigma)
        psf.select(snr_lim=snr_lim, dshift_lim=3, mask_lim=0.4, showme=True, nsig=30)
        psf.save(outname.replace('.fits',''))
        print('Final number of stars used:',np.sum(psf.ok))
        print(psf.psf_average.shape)
        psfmodel = renorm_psf(psf.psf_average, filt=pfilt, pixscl=img_pixel_scale, fov = pixsize*img_pixel_scale)
        fits.writeto('_'.join([outname.replace('.fits',''), 'psf_norm.fits']), np.array(psfmodel),overwrite=True)

        imshow(psf.data[psf.ok],nsig=50,title=psf.cat['id'][psf.ok])
        plt.savefig(outname.replace('.fits','_stamps_used.pdf').replace(outdir,plotdir),dpi=300)
        show_cogs([psf.psf_average],title=pfilt, label=['oPSF'],outname=plotdir+pfilt)
        plots=glob.glob(outdir+'*.pdf')
        plots+=glob.glob(outdir+'*_cat.fits')
        for plot in plots:
            os.rename(plot,plot.replace(outdir,plotdir))

        filt_psf = np.array(psf.psf_average)
        if pfilt == match_band:
            target_psf = filt_psf
        
        psfname = glob.glob(outdir+'*'+pfilt+'*'+'psf.fits')[0]
        outname = kernel_dir+os.path.basename(psfname).replace('psf','kernel')

        filt_psf = fits.getdata(psfname)
        if oversample > 1:
            print(f'Oversampling PSF by {oversample}x...')
            filt_psf = zoom(filt_psf, oversample)
            if pfilt == match_band:
                target_psf = zoom(target_psf, oversample)

        print(f'Normalizing PSF to unity...')
        filt_psf /= filt_psf.sum()

        if pfilt == match_band:
            target_psf /= target_psf.sum()
            continue
        
        if not os.path.exists(kernel_dir):
            os.mkdir(kernel_dir)

        print(f'Building {pfilt}-->{match_band} kernel...')
        assert filt_psf.shape == target_psf.shape, f'Shape of filter psf ({filt_psf.shape}) must match target psf ({target_psf.shape})'
        if method == 'pypher':
            fits.writeto(kernel_dir+'psf_a.fits',filt_psf,header=hdr,overwrite=True)
            fits.writeto(kernel_dir+'psf_b.fits',target_psf,header=hdr,overwrite=True)
            os.system(f'addpixscl {kernel_dir}psf_a.fits {img_pixel_scale}')
            os.system(f'addpixscl {kernel_dir}psf_b.fits {img_pixel_scale}')
            os.system(f'pypher {kernel_dir}psf_a.fits {kernel_dir}psf_b.fits {kernel_dir}kernel_a_to_b.fits -r {pypher_r:.3g}')
            kernel = fits.getdata(kernel_dir+'kernel_a_to_b.fits')
            os.remove(kernel_dir+'psf_a.fits')
            os.remove(kernel_dir+'psf_b.fits')
            os.remove(kernel_dir+'kernel_a_to_b.fits')
            os.remove(kernel_dir+'kernel_a_to_b.log')

        else:
            # Does a 2D cut around image - kinda like a tapered tophat
            # Only used if method is not pypher
            window = SplitCosineBellWindow(alpha=alpha, beta=beta)

            kernel =  create_matching_kernel(filt_psf, target_psf, window=window)

        if oversample > 1:
            kernel = block_reduce(kernel,block_size=oversample, func=np.sum)
            kernel /= kernel.sum()

        print(f'Writing {pfilt}-->{match_band} kernel to {outname}')
        fits.writeto(outname, np.float32(np.array(kernel/kernel.sum())),overwrite=True)

    nfilt = len(use_filters[1:])
    plt.figure(figsize=(30,nfilt*4))
    npanel = 7

    target_psf = fits.getdata(glob.glob(outdir+'*'+match_band+'*'+'psf.fits')[0])
    target_psf /= target_psf.sum()

    print(f'Plotting kernel checkfile...')
    for i, pfilt in enumerate(use_filters[1:]):
        # if pfilt.upper() not in ('F444W','F410M'): continue

        print(outdir)
        print(pfilt)
        psfname = glob.glob(outdir+'*'+pfilt+'*'+'psf.fits')[0]
        outname = kernel_dir+os.path.basename(psfname).replace('psf','kernel')

        filt_psf = fits.getdata(psfname)
        filt_psf /= filt_psf.sum()

        kernel = fits.getdata(outname)

        simple = simple_norm(kernel,stretch='linear',power=1, min_cut=-5e-4, max_cut=5e-4)

        plt.subplot(nfilt,npanel,1+i*npanel)
        plt.title('psf '+pfilt)
        plt.imshow(filt_psf, norm=simple, interpolation='antialiased',origin='lower')
        plt.subplot(nfilt,npanel,2+i*npanel)
        plt.title('target psf '+target_filter)
        plt.imshow(target_psf, norm=simple, interpolation='antialiased',origin='lower')
        plt.subplot(nfilt,npanel,3+i*npanel)
        plt.title("kernel "+pfilt)

        plt.imshow(kernel, norm=simple, interpolation='antialiased',origin='lower')

        filt_psf_conv = convolve_fft(filt_psf, kernel)

        plt.subplot(nfilt,npanel,4+i*npanel)
        plt.title("convolved "+pfilt)
        plt.imshow(filt_psf_conv, norm=simple, interpolation='antialiased',origin='lower')

        plt.subplot(nfilt,npanel,5+i*npanel)
        plt.title("residual "+pfilt)
        res = filt_psf_conv-target_psf
        plt.imshow(res, norm=simple, interpolation='antialiased',origin='lower')

        plt.subplot(nfilt,npanel,7+i*npanel)
        r,pf,pt = plot_profile(filt_psf_conv,target_psf)
        plt.plot(r*img_pixel_scale, pf/pt)
        fr,fpf,fpt = plot_profile(filt_psf,target_psf)
        plt.plot(fr*img_pixel_scale, fpf/fpt)
        plt.ylim(0.95,1.05)
        if method == 'pypher':
            plt.title('pypher r={}'.format(pypher_r))
        else:
            plt.title('alpha={}, beta={}'.format(ALPHA, BETA))
        plt.axvline(x=0.16,ls=':')
        plt.axhline(y=1,ls=':')
        plt.xlabel('radius arcsec')
        plt.ylabel('ee_psf_conv / ee_psf_target')

        plt.subplot(nfilt,npanel,6+i*npanel)
        plt.title('COG / COG_target')
        plt.plot(r*img_pixel_scale,pf,lw=3)
        plt.plot(r*img_pixel_scale,pt,'--',alpha=0.7,lw=3)
        plt.xlabel('radius arcsec')
        plt.ylabel('ee')

    plt.tight_layout()
    plt.savefig(plotdir+'kernels.pdf',dpi=300)

def renorm_psf(psfmodel, filt, fov=4.04, pixscl=0.03):
    '''
    Renormalize the PSF model to account for missing flux.

    Parameters:
    psfmodel (array): The PSF model.
    filt (str): The filter for which the PSF model is being generated.
    fov (float): The field of view (FOV) of the PSF in arcseconds.
    pixscl (float): The pixel scale of the PSF model in arcseconds per pixel.

    Returns:
    array: The renormalized PSF model.

    '''
    
    filt = filt.upper()

    # Encircled energy for WFC3 IR within 2" radius, ACS Optical, and UVIS from HST docs
    encircled = {}
    encircled['F225W'] = 0.993
    encircled['F275W'] = 0.984
    encircled['F336W'] = 0.9905
    encircled['F435W'] = 0.979
    encircled['F606W'] = 0.975
    encircled['F775W'] = 0.972
    encircled['F814W'] = 0.972
    encircled['F850LP'] = 0.970
    encircled['F098M'] = 0.974
    encircled['F105W'] = 0.973
    encircled['F125W'] = 0.969
    encircled['F140W'] = 0.967
    encircled['F160W'] = 0.966
    encircled['F090W'] = 0.9837
    encircled['F115W'] = 0.9822
    encircled['F150W'] = 0.9804
    encircled['F200W'] = 0.9767
    encircled['F277W'] = 0.9691
    encircled['F356W'] = 0.9618
    encircled['F410M'] = 0.9568
    encircled['F444W'] = 0.9546
    
    # These taken from the Encircled_Energy ETC numbers
    encircled['F140M'] = 0.984
    encircled['F162M'] = 0.982
    encircled['f182M'] = 0.980
    encircled['F210M'] = 0.978
    encircled['F250M'] = 0.971
    encircled['F300M'] = 0.967
    encircled['F335M'] = 0.963
    encircled['F360M'] = 0.958
    encircled['F430M'] = 0.956

    # Normalize to correct for missing flux
    # Has to be done encircled! Ensquared were calibated to zero angle...
    w, h = np.shape(psfmodel)
    Y, X = np.ogrid[:h, :w]
    r = fov / 2. / pixscl
    center = [w/2., h/2.]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    psfmodel /= np.sum(psfmodel[dist_from_center < r])
    if filt in encircled:
        psfmodel *= encircled[filt] # to get the missing flux accounted for
    else:
        print(f'WARNING -- I DO NOT HAVE ENCIRCLED ENERGY FOR {filt}! SKIPPING NORM.')

    return psfmodel

def imshow(args, cross_hairs=False, log=False, **kwargs):
    width = 20
    nargs = len(args)
    if nargs == 0: return
    if not (ncol := kwargs.get('ncol')): ncol = int(np.ceil(np.sqrt(nargs)))+1
    if not (nsig := kwargs.get('nsig')): nsig = 5
    if not (stretch := kwargs.get('stretch')): stretch = LinearStretch()

    nrow = int(np.ceil(nargs/ncol))
    panel_width = width/ncol
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol,figsize=(ncol*panel_width,nrow*panel_width))

    if type(ax) is not np.ndarray: ax = np.array(ax)
    for arg, axi in zip(args, ax.flat):
        sig = mad_std(arg[(arg != 0) & np.isfinite(arg)])
        if sig == 0: sig=1
        norm = ImageNormalize(np.float32(arg), vmin=-nsig*sig, vmax=nsig*sig, stretch=stretch)
        #axi.imshow(arg, norm=norm, origin='lower', cmap='gray',interpolation='nearest')
        axi.imshow(arg, norm=norm, origin='lower', interpolation='nearest')
        axi.set_axis_off()
        if cross_hairs:
            axi.plot(50,50, color='red', marker='+', ms=10, mew=1)
    #c = n/2
    #plt.plot([n/2,n/2],[n/2-2*r,n/2-r],c=color,lw=lw)
    #plt.plot([n/2-2*r,n/2-r],[n/2,n/2],c=color,lw=lw)
    #plot_cross_hairs(arg.shape[0],arg.shape[0]//4,color='red')

    if type(title := kwargs.get('title')) is not type(None):
        for fi,axi in zip(title,ax.flat): axi.set_title(fi)

    return fig, ax

class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'squareroot'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()

mscale.register_scale(SquareRootScale)


def find_stars(filenames=None, block_size=5, npeaks=1000, size=20, radii=np.array([0.5,1.,2.,4.,7.5]), range=[0,4], mag_lim = 24.0,
               threshold_min = -0.5, threshold_mode=[-0.2,0.2], shift_lim=2, zp=28.08, instars=None, showme=True, label='',
               outdir='./', plotdir='./'):

    if type(filenames) != list:
        filenames = [filenames]

    stars = []
    for filename in filenames:
        img, hdr = fits.getdata(filename, header=True)
        wcs = WCS(hdr)

        imgb = block_reduce(img, block_size, func=np.sum)
        sig = mad_std(imgb[imgb>0], ignore_nan=True)/block_size

        peaks = find_peaks(img, threshold=10*sig, npeaks=npeaks)
        # print(peaks)
        peaks.rename_column('x_peak','x')
        peaks.rename_column('y_peak','y')
        ra,dec = wcs.all_pix2world(peaks['x'], peaks['y'], 0)
        peaks['ra'] = ra
        peaks['dec'] = dec
        peaks['x0'] = 0.0
        peaks['y0'] = 0.0
        peaks['minv'] = 0.0
        for ir in np.arange(len(radii)): peaks['r'+str(ir)] = 0.
        for ir in np.arange(len(radii)): peaks['p'+str(ir)] = 0.

        t0 = time.time()
        
        for ip,p in enumerate(peaks):
            co = Cutout2D(img, (p['x'], p['y']), size, mode='partial')
            # measure offset, feed it to measure cog
            # if offset > 3 pixels -> skip
            position = centroid_com(co.data)
            peaks['x0'][ip] = position[0] - size//2
            peaks['y0'][ip] = position[1] - size//2
            peaks['minv'][ip] = np.nanmin(co.data)
            _ , cog, profile = measure_curve_of_growth(co.data, radii=np.array(radii), position=position, rnorm=None, rbg=None)
            for ir in np.arange(len(radii)): peaks['r'+str(ir)][ip] = cog[ir]
            for ir in np.arange(len(radii)): peaks['p'+str(ir)][ip] = profile[ir]
            co.radii = np.array(radii)
            co.cog = cog
            co.profile = profile
            stars.append(co)

    stars = np.array(stars)

    peaks['mag'] = zp-2.5*np.log10(peaks['r4'])
    r = peaks['r4']/peaks['r2']
    shift_lim_root = np.sqrt(shift_lim)

    ok_mag =  peaks['mag'] < mag_lim
    ok_min =  peaks['minv'] > threshold_min
    ok_phot = np.isfinite(peaks['r'+str(len(radii)-1)]) &  np.isfinite(peaks['r2']) & np.isfinite(peaks['p1'])
    ok_shift = (np.sqrt(peaks['x0']**2 + peaks['y0']**2) < shift_lim) & \
               (np.abs(peaks['x0']) < shift_lim_root) & (np.abs(peaks['y0']) < shift_lim_root)

    # ratio apertures @@@ hardcoded
    h = np.histogram(r[(r>1.2) & ok_mag], bins=np.arange(0, range[1], threshold_mode[1]/2.),range=range)
    ih = np.argmax(h[0])
    rmode = h[1][ih]
    ok_mode =  ((r/rmode-1) > threshold_mode[0]) & ((r/rmode-1) < threshold_mode[1])
    ok = ok_phot & ok_mode & ok_min & ok_shift & ok_mag
        
    # sigma clip around linear relation
    try:
        fitter = FittingWithOutlierRemoval(LinearLSQFitter(), sigma_clip, sigma=2.8, niter=2)
        lfit, outlier = fitter(Linear1D(),x=zp-2.5*np.log10(peaks['r4'][ok]),y=(peaks['r4']/peaks['r2'])[ok])
        ioutlier = np.where(ok)[0][outlier]
        ok[ioutlier] = False
    except:
        print('linear fit failed')
        ioutlier = 0
        lfit = None

    mags = zp-2.5*np.log10(peaks['r4'])

    peaks['id'] = 1
    peaks['id'][ok] = np.arange(1,len(peaks[ok])+1)

    if showme:
        if not os.path.exists(outdir): os.mkdir(outdir)
        plt.figure(figsize=(14,8))
        plt.subplot(231)
        mags = peaks['mag']
        mlim_plot = np.nanpercentile(mags,[5,95]) + np.array([-2,1])
        # print(mlim_plot)
        plt.scatter(mags,r,10)
        plt.scatter(mags[~ok_shift],r[~ok_shift],10,label='bad shift',c='C2')
        plt.scatter(mags[ok],r[ok],10,label='ok',c='C1')
        plt.scatter(mags[ioutlier],r[ioutlier],10,label='outlier',c='darkred')
        if lfit: plt.plot(np.arange(14,30), lfit(np.arange(14,30)),'--',c='k',alpha=0.3,label='slope = {:.3f}'.format(lfit.slope.value))
        plt.legend()
        plt.ylim(0,14)
        plt.xlim(mlim_plot[0],mlim_plot[1])
        plt.title(' aper(2) / aper(4) vs mag(aper(4))')

        plt.subplot(232)
        ratio_median = np.nanmedian(r[ok])
        plt.scatter(mags,r,10)
        plt.scatter(mags[~ok_shift],r[~ok_shift],10,label='bad shift',c='C2')
        plt.scatter(mags[ok],r[ok],10,label='ok',c='C1')
        plt.scatter(mags[ioutlier],r[ioutlier],10,label='outlier',c='darkred')
        if lfit: plt.plot(np.arange(15,30), lfit(np.arange(15,30)),'--',c='k',alpha=0.3,label='slope = {:.3f}'.format(lfit.slope.value))
        plt.legend()
        plt.ylim(ratio_median-1,ratio_median+1)
        plt.xlim(mlim_plot[0],mlim_plot[1])
        plt.title('aper(2) / aper(4) vs mag(aper(4))')

        plt.subplot(233)
        _ = plt.hist(r,bins=range[1]*20,range=range)
        _ = plt.hist(r[ok],bins=range[1]*20,range=range)
        plt.title('aper(2) / aper(4)')

        plt.subplot(234)
        plt.scatter(zp-2.5*np.log10(peaks['r3'][ok]),(peaks['peak_value']/peaks['r3'])[ok])
        plt.scatter(zp-2.5*np.log10(peaks['r3'])[ioutlier],(peaks['peak_value'] /peaks['r3'])[ioutlier],c='darkred')
        plt.ylim(0,1)
        plt.title('peak / aper(3) vs maper(3)')

        plt.subplot(235)
        plt.scatter(peaks['x0'][ok],peaks['y0'][ok],c='C1')
        plt.scatter(peaks['x0'][ioutlier],peaks['y0'][ioutlier],c='darkred')
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.title('offset (pix)')

        plt.subplot(236)
        plt.scatter(peaks['x'][ok],peaks['y'][ok],c='C1')
        plt.scatter(peaks['x'][ioutlier],peaks['y'][ioutlier],c='darkred')
        plt.axis('scaled')
        plt.title('position (pix)')
        plt.tight_layout()
        suffix = '.fits' + filename.split('.fits')[-1]
        plt.savefig(outdir+'/'+os.path.basename(filename).replace(suffix,'_diagnostic.pdf'))
        
        dd = [st.data for st in stars[ok]]
        title = ['{} {:.1f} {:.2f} {:.2f} {:.1f} {:.1f}'.format(ii, mm, pp,qq,xx,yy) for ii,mm,pp,qq,xx,yy in zip(peaks['id'][ok],mags[ok],peaks['p1'][ok],peaks['minv'][ok],peaks['x0'][ok],peaks['y0'][ok])]
        imshow(dd,nsig=30,title=title)
        plt.tight_layout()
        plt.savefig(outdir+'/'+os.path.basename(filename).replace(suffix,'_star_stamps.pdf'))
    
    peaks[ok].write(outdir+'/'+os.path.basename(filename).replace(suffix,'_star_cat.fits'),overwrite=True)
                
    return peaks[ok], stars[ok]

class PSF():
    def __init__(self, images=None, all_x=None, all_y=None, all_ids=None, pixsize=101, pixelscale=0.03):
        
        if type(images) != list:
            images = [images]
        
        assert len(images) == len(all_x) == len(all_y) == len(all_ids)

        cats, datas = [], []
        for i, image in enumerate(images):
            x = all_x[i]
            y = all_y[i]
            ids = all_ids[i]
            
            if type(image) == np.ndarray:
                img = image
                xx=x[i]
                yy=y[i]
                
                self.filename = None
            else:
                img, hdr = fits.getdata(image, header=True)
                wcs = WCS(hdr)
                xx,yy = wcs.all_world2pix(x, y, 0)
                self.filename  = images

            if type(ids) == type(None):
                ids = np.arange(1,len(x)+1)

            self.nx = pixsize
            self.c0 = self.nx//2
            #print([ids,xx,yy,x,y, i*np.ones(len(ids))])
            
            cat = Table([ids,xx,yy,x,y, i*np.ones(len(xx))],names=['id','x','y','ra','dec', 'img_pos'])
            cats.append(cat)

            data = np.array([Cutout2D(img, (xx[i],yy[i]), (pixsize, pixsize),mode='partial').data for i in np.arange(len(x))])
            datas.append(data)

        if len(cats) > 1:
            self.cat = vstack(cats)
        else:
            self.cat = cats[0]
        
        self.data = np.ndarray.flatten(np.array(datas))
        self.data = np.ma.array(data,mask = ~np.isfinite(data) | (data == 0) )
        self.data_orig = self.data.copy()
        self.ok = np.ones(len(self.cat))

    def phot(self, radius=8):
        pos = np.array([self.cat['x'],self.cat['y']])

        caper = CircularAperture((self.c0,self.c0), r=radius)
        cmask = Cutout2D(caper.to_mask(),(radius,radius), self.nx,mode='partial').data   # check ding galfit for better way
        phot = [aperture_photometry(st, caper)['aperture_sum'][0] for st in self.data]
        return phot

    def measure(self,norm_radius=8):
        peaks = []
        c0 = self.nx//2
        self.norm_radius = norm_radius

        peaks = np.array([st.max()for st in self.data])
        peaks[~np.isfinite(peaks) | (peaks==0)] = 0
        pos = np.array([self.cat['x'],self.cat['y']])

        #data = np.array(self.data)
        #data[self.data.mask] = 0
        caper = CircularAperture((self.c0,self.c0), r=self.norm_radius)
        cmask = Cutout2D(caper.to_mask(),(self.norm_radius,self.norm_radius), self.nx,mode='partial').data

        phot = self.phot(radius=self.norm_radius)
        sat =  [aperture_photometry(st, caper)['aperture_sum'][0] for st in np.array(self.data)] # casting to array removes mask
        cmin = [ np.nanmin(st*cmask) for st in self.data]

        self.cat['frac_mask'] = 0.0

        for i in np.arange(len(self.data)):
            self.data[i].mask |= (self.data[i]*cmask) < 0.0

        self.cat['peak'] =  peaks
        self.cat['cmin'] =  np.array(cmin)
        self.cat['phot'] =  np.array(phot)
        self.cat['saturated'] =  np.int32(~np.isfinite(np.array(sat)))

        flip_ratio = []
        rms_array = []
        for st in self.data:
            rms, snr = stamp_rms_snr(st)
            rms_array.append(rms)
            dp = st - np.flip(st,axis=(0,1))
            flip_ratio.append(np.abs(st).sum()/np.abs(dp).sum())

        self.cat['snr'] = 2*np.array(phot)/np.array(rms_array)
        self.cat['flip_ratio'] = np.array(flip_ratio)
        self.cat['phot_frac_mask'] = 1.0

    def select(self, snr_lim = 800, dshift_lim=3, mask_lim=0.40, phot_frac_mask_lim = 0.85, showme=False, **kwargs):
        self.ok = (self.cat['dshift'] < dshift_lim) & (self.cat['snr'] > snr_lim) & (self.cat['frac_mask'] < mask_lim) & (self.cat['phot_frac_mask'] > phot_frac_mask_lim)

    #(self.cat['cmin'] >= -1.5)  #& (self.cat['cmin'] >= -1.5)
        self.cat['ok'] = np.int32(self.ok)
        self.cat['ok_shift'] = (self.cat['dshift'] < dshift_lim)
        self.cat['ok_snr'] = (self.cat['snr'] > snr_lim)
        self.cat['ok_frac_mask'] = (self.cat['frac_mask'] < mask_lim)
        self.cat['ok_phot_frac_mask'] = (self.cat['phot_frac_mask'] > phot_frac_mask_lim)

        for c in self.cat.colnames:
            if 'id' not in c: self.cat[c].format='.3g'

        if showme:
            title = f"{self.cat['id']}, {self.cat['ok']}"
            fig, ax = imshow(self.data, title=title,**kwargs)
            fig.savefig('test.pdf',dpi=300)
            # self.cat.pprint_all()


    def stack(self,sigma=3,maxiters=2):
        iok = np.where(self.ok)[0]

        norm = self.cat['phot'][iok]
        data = self.data_orig[iok].copy()
        for i in np.arange(len(data)): data[i] = data[i]/norm[i]

        stack, lo, hi, clipped = sigma_clip_3d(data,sigma=sigma,axis=0,maxiters=maxiters)
        self.clipped = clipped
       # self.clipped[~np.isfinite(self.clipped)] = 0

        # print('-',len(self.ok[self.ok]))

        for i in np.arange(len(data)):
            self.ok[iok[i]] = self.ok[iok[i]] and ~self.clipped[i].mask[50,50]
            self.data[iok[i]].mask = self.clipped[i].mask
            mask = self.data[iok[i]].mask
            self.cat['frac_mask'][iok[i]] = np.size(mask[mask]) / np.size(mask)

        self.psf_average = stack
        self.cat['phot_frac_mask'] = self.phot(radius=self.norm_radius)/self.cat['phot']

    def show_by_id(self, ID, **kwargs):
        indx = np.where(self.cat['id']==ID)[0][0]
        imshow([self.data[indx]], **kwargs)
        return self.data[indx]

    def growth_curves(self):
        for i in np.arange(len(self.data)):
            radii, cog, profile = measure_curve_of_growth(a)
            r = radii*self.pixelscale

    def center(self,window=21,interpolation=cv2.INTER_CUBIC):
        if 'x0' in self.cat.colnames: return

        cw = window//2
        c0 = self.c0
        pos = []
        for i in np.arange(len(self.data)):
            p = self.data[i,:,:]
            st = Cutout2D(p,(self.c0,self.c0),window,mode='partial',fill_value=0).data
            st[~np.isfinite(st)] = 0
            x0, y0 = centroid_com(st)

            p = imshift(p, (cw-x0), (cw-y0),interpolation=interpolation)

            # now measure shift on recentered cutout
            # first in small window
            st = Cutout2D(p,(self.c0,self.c0),window,mode='partial',fill_value=0).data
            x1,y1 = centroid_com(st)

            # now in central half of stamp
            st2 = Cutout2D(p,(self.c0,self.c0),int(self.nx*0.5),fill_value=0).data
            # measure moment shift in positive definite in case there are strong ying yang residuals
            x2,y2 = centroid_com(np.maximum(p,0))

            p = np.ma.array(p, mask = ~np.isfinite(p) | (p==0))
            self.data[i,:,:] = p

            # difference in shift between central window and half of stamp is measure of contamination
            # from bright off axis sources
            dsh = np.sqrt(((c0-x2)-(cw-x1))**2 + ((c0-y2)-(cw-y1))**2)
            pos.append([cw-x0,cw-y0,cw-x1,cw-y1,dsh])

        self.cat = hstack([self.cat,Table(np.array(pos),names=['x0','y0','x1','y1','dshift'])])

    def save(self, outname=''):
        # with open('_'.join([outname, 'psf_stamps.fits']), 'wb') as handle:
        #     self.data[self.ok]

        # self.data[self.ok].filled(-99).write('_'.join([outname, 'psf_stamps.fits']),overwrite=True)

        fits.writeto('_'.join([outname, 'psf.fits']), np.array(self.psf_average),overwrite=True)

        self.cat[self.ok].write('_'.join([outname, 'psf_cat.fits']),overwrite=True)

        title = f"{self.cat['id']}, {self.cat['ok']}"
        fig, ax = imshow(self.data, nsig=30, title=title)
        fig.savefig('_'.join([outname, 'psf_stamps.pdf']),dpi=300)

def measure_curve_of_growth(image, position=None, radii=None, rnorm='auto', nradii=30, verbose=False, showme=False, rbg='auto'):
    """
    Measure a curve of growth from cumulative circular aperture photometry on a list of radii centered on the center of mass of a source in a 2D image.

    Parameters
    ----------
    image : `~numpy.ndarray`
        2D image array.
    position : `~astropy.coordinates.SkyCoord`
        Position of the source.
    radii : `~astropy.units.Quantity` array
        Array of aperture radii.

    Returns
    -------
    `~astropy.table.Table`
        Table of photometry results, with columns 'aperture_radius' and 'aperture_flux'.
    """

    if type(radii) is type(None):
        radii = powspace(0.5,image.shape[1]/2,nradii)

    # Calculate the centroid of the source in the image
    if type(position) is type(None):
        #x0, y0 = centroid_2dg(image)
        position = centroid_com(image)

    if rnorm == 'auto': rnorm = image.shape[1]/2.0
    if rbg == 'auto': rbg = image.shape[1]/2.0

    apertures = [CircularAperture(position, r=r) for r in radii]

    if rbg:
        bg_mask = apertures[-1].to_mask().to_image(image.shape) == 0
        bg = np.nanmedian(image[bg_mask])
        if verbose: print('background',bg)
    else:
        bg = 0.

    # Perform aperture photometry for each aperture
    phot_table = aperture_photometry(image-bg, apertures)
    # Calculate cumulative aperture fluxes
    cog = np.array([phot_table['aperture_sum_'+str(i)][0] for i in range(len(radii))])

    if rnorm:
        rnorm_indx = np.searchsorted(radii, rnorm)
        cog /= cog[rnorm_indx]


    area = np.pi*radii**2
    area_cog = np.insert(np.diff(area),0,area[0])
    profile = np.insert(np.diff(cog),0,cog[0])/area_cog
    profile /= profile.max()

    if showme:
        plt.plot(radii, cog, marker='o')
        plt.plot(radii,profile/profile.max())
        plt.xlabel('radius pix')
        plt.ylabel('curve of growth')

    # Create output table of aperture radii and cumulative fluxes
    return radii, cog, profile

def imshift(img, ddx, ddy, interpolation=cv2.INTER_CUBIC):
    # affine matrix
    M = np.float32([[1,0,ddx],[0,1,ddy]])
    # output shape
    wxh = img.shape[::-1]
    return cv2.warpAffine(img, M, wxh, flags=interpolation)

def show_cogs(*args, title='', linear=False, pixscale=0.03, label=None, outname=''):
    npsfs = len(args)
    nfilts = len(args[0])

    xtick = [0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0]
    plt.figure(figsize=(20,4.5))

    if not label:
        label = ['' for p in range(npsfs)]

    for filti in range(nfilts):
        psf_ref = args[0][filti]
        psf_ref2 = args[-1][filti]
        r, cog_ref, prof_ref = measure_curve_of_growth(psf_ref,nradii=50)
        r, cog_ref2, prof_ref2 = measure_curve_of_growth(psf_ref2,nradii=50)
        r = r * pixscale

        plt.subplot(141)
        plt.plot(r,prof_ref,label=label[0])
        plt.title(title+' profile')
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
        plt.yscale('log')
        plt.xlim(0,1)
        plt.ylim(1e-5,1)
        plt.xlabel('arcsec')
        plt.axhline(y=0,alpha=0.5,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        ax=plt.gca()
        rms, snr = stamp_rms_snr(psf_ref)
        dx, dy = centroid_com(psf_ref)
        plt.text(0.6,0.8,'snr = {:.2g} \nx0,y0 = {:.2f},{:.2f} '.format(snr,dx,dy),transform=ax.transAxes, c='C0')

        plt.subplot(142)

        plt.plot(r,cog_ref,label=label[0])
        plt.xlabel('arcsec')
        plt.title('cog')
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
        plt.axhline(y=1,alpha=0.3,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.08,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.04,alpha=0.5,c='k',ls='--')
        plt.xlim(0.02,1)

        plt.subplot(143)
        plt.plot(r,np.ones_like(r),label=label[0])
        plt.xlabel('arcsec')
        plt.title('cog / cog_'+label[0])
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
        plt.xlabel('arcsec')
        plt.axhline(y=1,alpha=0.3,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.08,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.04,alpha=0.5,c='k',ls='--')
        plt.xlim(0.02,1)
        plt.ylim(0.5,1.5)

        plt.subplot(144)
        plt.plot(r,cog_ref/cog_ref2,label=label[0],c='C0')
        plt.xlabel('arcsec')
        plt.title('cog / cog_'+label[-1])
        if not linear:
            plt.xscale('squareroot')
            plt.xticks(xtick)
        plt.xlabel('arcsec')
        plt.axhline(y=1,alpha=0.3,c='k')
        plt.axvline(x=0.16,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.08,alpha=0.5,c='k',ls='--')
        plt.axvline(x=0.04,alpha=0.5,c='k',ls='--')
        plt.xlim(0.02,1)
        plt.ylim(0.5,1.5)


        cogs = []
        profs = []
        psfs = [psf_ref]
        for psfi in np.arange(1, npsfs):
            psf = args[psfi][filti]
            _, cog, prof = measure_curve_of_growth(psf,nradii=50)
            cogs.append(cog)
            profs.append(prof)
            dx, dy = centroid_com(psf)
            rms, snr = stamp_rms_snr(psf)

            plt.subplot(141)
            plt.plot(r,prof)

            plt.text(0.5,0.8-psfi*0.1,'snr = {:.2g} \nx0,y0 = {:.2f},{:.2f} '.format(snr,dx,dy),transform=ax.transAxes, c='C'+str(psfi))
            plt.xlim(0.02,1)

            plt.subplot(142)
            plt.plot(r,cog,label=label[psfi],c='C'+str(psfi))
            plt.legend()

            plt.subplot(143)
            plt.plot(r,cog/cog_ref,c='C'+str(psfi))

            plt.subplot(144)
            plt.plot(r,cog/cog_ref2,c='C'+str(psfi))

            psfs.append(psf)

        plt.savefig('_'.join([outname,'psf_cog.pdf']),dpi=300)

        _ = imshow(psfs,cross_hairs=True,nsig=50,title=label)

        plt.savefig('_'.join([outname,'psf_average.pdf']),dpi=300)

def stamp_rms_snr(img, block_size=3, rotate=True):
    if rotate:
        p180 = np.flip(img,axis=(0,1))
        dp = img-p180
    else:
        dp = img.copy()

    s = dp.shape[1]
    buf = 6
    dp[s//buf:(buf-1)*s//buf,s//buf:(buf-1)*s//buf] = np.nan
    dp3 = block_reduce(dp,block_size=3)

    rms = mad_std(dp,ignore_nan=True)/block_size * np.sqrt(img.size)
    if rotate: rms /= np.sqrt(2)

    snr = img.sum()/rms

    return rms, snr

def sigma_clip_3d(data, maxiters=2, axis=0, **kwargs):
    clipped_data = data.copy()
    for i in range(maxiters):
        clipped_data, lo, hi = sigma_clip(clipped_data, maxiters=0, axis=0, masked=True, grow=False, return_bounds=True, **kwargs)
        # grow mask
        for i in range(len(clipped_data.mask)): clipped_data.mask[i,:,:] = grow(clipped_data.mask[i,:,:],iterations=1)

    return np.mean(clipped_data,axis=axis), lo, hi, clipped_data

def grow(mask, structure=disk(2), **kwargs):
    return binary_dilation(mask,structure=structure,**kwargs)

def powspace(start, stop, num=30, power=0.5, **kwargs):
    """Generate a square-root spaced array with a specified number of points
    between two endpoints.

    Parameters
    ----------
    start : float
        The starting value of the range.
    stop : float
        The ending value of the range.
    pow: power of distribution, defaults to sqrt
    num_points : int, optional
        The number of points to generate in the array. Default is 50.

    Returns
    -------
    numpy.ndarray
        A 1-D array of `num_points` values spaced equally in square-root space
        between `start` and `stop`.
    """
    return np.linspace(start**power, stop**power, num=num, **kwargs)**(1/power)

def plot_profile(psf, target):
    shape = psf.shape
    center = (shape[1]//2, shape[0]//2)
    radii_pix = np.arange(1,40,1)
    apertures = [CircularAperture(center, r=r) for r in radii_pix] #r in pixels

    phot_table = aperture_photometry(psf, apertures)
    flux_psf = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])

    phot_table = aperture_photometry(target, apertures)
    flux_target = np.array([phot_table[0][3+i] for i in range(len(radii_pix))])

    return radii_pix[:-1], (flux_psf)[0:-1], (flux_target)[0:-1]


def convolve_images(bands, im_paths, wht_paths, err_paths, outdir, kernel_dir, match_band, use_fft_conv=True, overwrite=False):
    #kernel = fits.getdata(f'{kernel_dir}{match_band}_kernel.fits')
    
    if use_fft_conv:
        convolve_func = convolve_fft
        convolve_kwargs = {'allow_huge': True}
    else:
        convolve_func = convolve
        convolve_kwargs = {}

    #for filename in sci_paths:
    for band in bands:
        im_filename = im_paths[band]
        wht_filename = wht_paths[band]
        err_filename = err_paths[band]
        same_file = im_filename == wht_filename == err_filename
        outnames = []

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if same_file:
            print('WHT, SCI, and ERR are the same file!. Output will be written to the same file.')
            outname = im_filename.replace('.fits', f'_{match_band}-matched.fits').replace(os.path.dirname(im_filename), outdir)
            outnames.append(outname)
            outsciname = outwhtname = outerrname = outname
        else:
            outsciname = im_filename.replace(f'.fits', f'_sci_{match_band}-matched.fits').replace(os.path.dirname(im_filename), outdir)
            outwhtname = wht_filename.replace(f'.fits', f'_wht_{match_band}-matched.fits').replace(os.path.dirname(wht_filename), outdir)
            outerrname = err_filename.replace(f'.fits', f'_err_{match_band}-matched.fits').replace(os.path.dirname(err_filename), outdir)
            outnames.append(outsciname)
            outnames.append(outwhtname)
            outnames.append(outerrname)

        skip = False
        for outname in outnames:    
            if os.path.exists(outname) and not overwrite:
                print(outsciname, outwhtname)
                print(f'Convolved images exist, I will not overwrite')
                skip = True
        
        if skip:
            continue

        print('  science image: ', im_filename)
        print('  weight image: ', wht_filename)
        print('  error image: ', err_filename)
        hdul = fits.open(im_filename)
        
        hdul_wht = fits.open(wht_filename)
        if err_filename != '':
            hdul_err = fits.open(err_filename)

        if band != match_band:
            print(f'  PSF-matching sci {band} to {match_band}')
            tstart = time.time()
            fn_kernel = os.path.join(kernel_dir, f'{band}_kernel.fits')
            print('  using kernel ', fn_kernel.split('/')[-1])
            kernel = fits.getdata(fn_kernel)
            kernel /= np.sum(kernel)

            if same_file:
                wht_ext = 'WHT'
            else:
                wht_ext = 0
            weight = hdul_wht[wht_ext].data
            
            out_hdul = fits.HDUList([])
            
            if overwrite or not os.path.exists(outsciname):
                print('Running science image convolution...')
                if same_file:
                    sci_ext = 'SCI'
                else:
                    sci_ext = 0
                
                sci = hdul[sci_ext].data
                data = convolve_func(sci, kernel, **convolve_kwargs).astype(np.float32)
                data[weight==0] = 0.
                print('convolved...')

                out_hdu = fits.PrimaryHDU(data, header=hdul[sci_ext].header)
                out_hdu.name = 'SCI'
                out_hdu.header['HISTORY'] = f'Convolved with {match_band} kernel'
                out_hdul.append(out_hdu)

                if not same_file:
                    out_hdul.writeto(outsciname, overwrite=True)
                    print('Wrote file to ', outsciname)
                    out_hdul = fits.HDUList([])
               
                
            else:
                print(outsciname)
                print(f'{band.upper()} convolved science image exists, I will not overwrite')
            
            hdul.close()

            if overwrite or not os.path.exists(outwhtname):
                print('Running weight image convolution...')
                err = np.where(weight==0, 0, 1/np.sqrt(weight))
                err_conv = convolve_func(err, kernel, **convolve_kwargs).astype(np.float32)
                data = np.where(err_conv==0, 0, 1./(err_conv**2))
                data[weight==0] = 0.

                out_hdu_wht = fits.PrimaryHDU(data, header=hdul_wht[wht_ext].header)
                out_hdu_wht.name = 'WHT'
                out_hdu_wht.header['HISTORY'] = f'Convolved with {match_band} kernel'

                out_hdul.append(out_hdu_wht)
                if not same_file:
                    out_hdul.writeto(outwhtname, overwrite=True)
                    print('Wrote file to ', outwhtname)
                    out_hdul = fits.HDUList([])
                
            else:
                print(outwhtname)
                print(f'{band.upper()} convolved weight image exists, I will not overwrite')
            
            hdul_wht.close()

            if outerrname != '' and (overwrite or not os.path.exists(outerrname)):
                print('Running error image convolution...')
                
                data = convolve_func(hdul_err['ERR'].data, kernel, **convolve_kwargs).astype(np.float32)
                data[weight==0] = 0.

                out_hdu_err = fits.PrimaryHDU(data, header=hdul_err['ERR'].header)
                out_hdu_err.name = 'ERR'
                out_hdu_err.header['HISTORY'] = f'Convolved with {match_band} kernel'
                out_hdul.append(out_hdu_err)
                if not same_file:
                    
                    out_hdul.writeto(outerrname, overwrite=True)
                    print('Wrote file to ', outerrname)
                    out_hdul = fits.HDUList([])
                

                hdul_err.close()

            print(f'Finished in {time.time()-tstart:2.2f}s')

            if same_file and len(out_hdul) > 1:
                out_hdul.writeto(outname, overwrite=True)
            else:
                print('Not writing empty HDU')
        else:

            outsciname = im_filename.replace(f'.fits', f'_sci_{match_band}-matched.fits').replace(os.path.dirname(im_filename), outdir)
            outwhtname = wht_filename.replace(f'.fits', f'_wht_{match_band}-matched.fits').replace(os.path.dirname(wht_filename), outdir)
            outerrname = err_filename.replace(f'.fits', f'_err_{match_band}-matched.fits').replace(os.path.dirname(err_filename), outdir)
            outname = im_filename.replace(f'.fits', f'_{match_band}-matched.fits').replace(os.path.dirname(im_filename), outdir)
            
            if same_file:
                hdul.writeto(outname, overwrite=True)
                print(hdul.info())
            else:
                hdul.writeto(outsciname, overwrite=True)
                hdul_wht.writeto(outwhtname, overwrite=True)
                if err_filename != '':
                    hdul_err.writeto(outerrname, overwrite=True)
            
            print('Written files to ', outname)
                

def psf_comparison(bands, psf_dir_dict, max_cols=5, cmap = 'cmr.torch', match_band='F444W', pixelscale=0.03):
    try:
        plt.style.use('/nvme/scratch/work/tharvey/scripts/paper.mplstyle')
        print('Using custom style')
    except FileNotFoundError:
        pass
    nrows = int(np.ceil(len(bands)/max_cols))
    colors = plt.get_cmap(cmap)
    band_colors = {band:colors(i/len(bands)) for i, band in enumerate(bands)}
    tool_colors = {key:colors(i/len(psf_dir_dict)) for i, key in enumerate(psf_dir_dict.keys())}
    linestyles = ['-', '--', '-.', ':']
    fig, axs = plt.subplots(nrows = nrows, ncols = max_cols, figsize=(max_cols*2, nrows*2), sharex=True, sharey=True, constrained_layout=True)
    axs = axs.flatten()
    # Remove empty axes
    for i in range(len(bands), len(axs)):
        fig.delaxes(axs[i])
    
    #assert len(kernel_dir_dict.keys()) == len(psf_dir_dict.keys()), 'PSF and kernel names must match'

    for i, band in enumerate(bands):


        for j, (name, psf_dir) in enumerate(psf_dir_dict.items()):
            #filename = glob.glob(f'{kernel_dir}/*{band}*.fits')[0]
            #kernel = fits.getdata(filename)
            #psf_dir = psf_dir_dict[name]
            filename = glob.glob(f'{psf_dir}/*{band}*.fits')
            if len(filename) > 1:
                try:
                    filename = [f for f in filename if 'orig' not in f]
                    if len(filename) > 1:
                        filename = [f for f in filename if 'norm' in f]
                    filename = filename[0]
                except:
                    filename = filename[0]
            elif len(filename) == 0:
                print(f'No PSF found for {band} in {psf_dir}')
                continue
            else:
                filename = filename[0]
            psf = fits.getdata(filename)
           
            # nradii should depend on dimensions of PSF
            nradii = int(np.sqrt(np.sum(np.array(psf.shape)**2))/2)
            
            radii, cog, profile = measure_curve_of_growth(psf, rnorm=None, nradii=nradii)
            
            
            radii = radii*pixelscale if type(pixelscale) == float else radii*pixelscale[name]
            # Interpolate COG to get radii at COG = 0.8
            cog_interp = scipy.interpolate.interp1d(cog, radii, fill_value='extrapolate')
            nearrad = cog_interp(0.8)
            # Convert 
            axs[i].plot([nearrad, nearrad], [0, 0.8], c=tool_colors[name], alpha=0.9, linestyle='--', lw=1)

            axs[i].plot(radii, cog, label=name, c=tool_colors[name], linestyle='solid', lw = 1.5, alpha=0.8)

        #axs[i].set_title(band)
        # Remove whitspace between axis

        axs[i].text(0.8, 0.08, band, ha='center', va='center', transform=axs[i].transAxes, fontsize=12)
        #axs[i].axhline(1, c='k', alpha=0.3)
        axs[i].axvline(0.16, c='k', alpha=0.3, linestyle='--', lw=1)
        #axs[i].axhline(0.99, c='k', alpha=0.3, linestyle=':', lw=2)
        axs[i].set_ylim(0.3, 1.)
        axs[i].set_xlim(0.05, 1.)
        axs[i].set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        
        fig.supxlabel('Radius (arcsec)', fontsize=14)
        fig.supylabel('Encircled Energy', fontsize=14)
        # Add ticks to upper axis
        axs[i].xaxis.set_tick_params(which='both', top=True)

        if i == len(bands)-1:
            
            axs[i].legend(frameon=False, fontsize=7, loc='lower right', bbox_to_anchor=(1.0, 0.13))
            axs[i].text(0.17, 0.84, '0.16 as', ha='center', va='center', transform=axs[i].transAxes, fontsize=8, rotation=90, alpha=0.8)
    fig.get_layout_engine().set(wspace=-5, hspace=-3)

    fig.savefig('/nvme/scratch/work/tharvey/PSFs/psf_comparison.pdf', dpi=300)    


def measure_cog(sci_cutout, pos):
    # make COG by apertures
    radii = np.arange(1, np.shape(sci_cutout)[0]/2., 0.1)
    apertures = [CircularAperture(pos, r) for r in radii]
    phot_tab = aperture_photometry(sci_cutout, apertures)
    cog = np.array([[phot_tab[coln][0] for coln in phot_tab.colnames[3:]]][0])

    return radii, cog

# Compute COG for PSF
def psf_cog(psfmodel, filt, nearrad=None, fix_extrapolation=True, pixel_scale=None, norm_rad=1.0):
    pos = np.shape(psfmodel)[0]/2.,  np.shape(psfmodel)[1]/2.
    radii, cog = measure_cog(psfmodel, pos)
    radii *= pixel_scale

    if fix_extrapolation:
        
        SW_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
        LW_FILTERS = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                    'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                    'F466N', 'F470N', 'F480M']

        PATH_SW_ENERGY = '/nvme/scratch/work/tharvey/aperpy/harvey_testing/PSF/Encircled_Energy_SW_ETCv2.txt'
        PATH_LW_ENERGY = '/nvme/scratch/work/tharvey/aperpy/harvey_testing/PSF/Encircled_Energy_LW_ETCv2.txt'

        from astropy.io import ascii
        # Check if filter is valid and get correction term
        if filt in SW_FILTERS:
            encircled = ascii.read(PATH_SW_ENERGY)
        elif filt in LW_FILTERS:
            encircled = ascii.read(PATH_LW_ENERGY)
        else:
            print(f'{filt} is NOT a valid NIRCam filter!')
            return

        # max_rad = radius[-1]
        large_rad = encircled['aper_radius']
        large_ee =  encircled[filt]

        ok = radii<norm_rad
        radii = radii[ok]
        cog = cog[ok]
        cog *= large_ee[np.argmin(np.abs(large_rad-norm_rad))] / cog[-1]
        radii = np.array(list(radii)+list(large_rad[large_rad>norm_rad]))
        cog_norm = np.array(list(cog)+list(large_ee[large_rad>norm_rad]))

    import scipy.interpolate
    modcog_norm = scipy.interpolate.interp1d(radii, cog_norm, fill_value = 'extrapolate')

    if nearrad is None:
        return radii, cog_norm, modcog_norm

    else:
        output = modcog_norm(nearrad)
        output[output>1] = 1.0
        return output


# Make and rotate PSF
def get_webbpsf(filt, field='uncover', angle=None, fov=4, og_fov=10, pixscl=None, date=None, output='', jitter_kernel='gaussian', jitter_sigma=None):
    # makes the PSF at og_fov and clips down to fov. Tested with 0.04 "/px
    import webbpsf
    from astropy.io import fits
    import numpy as np
    from scipy import ndimage
    import os
    from astropy.io import ascii

    #from config import SW_FILTERS, LW_FILTERS, PATH_SW_ENERGY, PATH_LW_ENERGY
    SW_FILTERS = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N',
                'F150W2', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N']
    LW_FILTERS = ['F250M', 'F277W', 'F300M', 'F322W2', 'F323N', 'F335M',
                'F360M', 'F356W', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M',
                'F466N', 'F470N', 'F480M']
    PATH_SW_ENERGY = '/nvme/scratch/work/tharvey/aperpy/config/Encircled_Energy_SW_ETCv2.txt'
    PATH_LW_ENERGY = '/nvme/scratch/work/tharvey/aperpy/config/Encircled_Energy_LW_ETCv2.txt'


    print(f'{filt} at {fov}" FOV')

    if pixscl is None:
        from config import PIXEL_SCALE
        pixscl = PIXEL_SCALE

    # Check if filter is valid and get correction term
    if filt in SW_FILTERS:
        detector = 'NCRA5'
        # 17 corresponds with 2" radius (i.e. 4" FOV)
        energy_table = ascii.read(PATH_SW_ENERGY)
        row = np.argmin(abs(fov/2. - energy_table['aper_radius']))
        encircled = energy_table[row][filt]
        norm_fov = energy_table['aper_radius'][row] * 2
        print(f'Will normalize PSF within {norm_fov}" FOV to {encircled}')
    elif filt in LW_FILTERS:
        detector = 'NCRA1'
        energy_table = ascii.read(PATH_LW_ENERGY)
        row = np.argmin(abs(fov/2. - energy_table['aper_radius']))
        encircled = energy_table[row][filt]
        norm_fov = energy_table['aper_radius'][row] * 2
        print(f'Will normalize PSF within {norm_fov}" FOV to {encircled}')
    else:
        print(f'{filt} is NOT a valid NIRCam filter!')
        return

    # Observed PA_V3 for fields
    angles = {'ceers': 130.7889803307112, 'smacs': 144.6479834976019, 'glass': 251.2973235468314, 'uncover': 41.3,
          'primer-cosmos': 292., 'primer-uds': 70.}
    if angle is None:
        angle = angles[field]
    nc = webbpsf.NIRCam()
    nc.options['parity'] = 'odd'
    if not os.path.exists(output):
        os.makedirs(output)
    outname = os.path.join(output, 'psf_'+field+'_'+filt+'_'+str(fov)+'arcsec_'+str(angle)) # what to save as?

    if jitter_sigma is not None:
        nc.options['jitter'] = jitter_kernel   # jitter model name or None
        nc.options['jitter_sigma'] = jitter_sigma  # in arcsec per axis, default 0.007
        if jitter_kernel == 'gaussian':
            outname += f'_jitter{int(jitter_sigma*1000)}mas'


    # make an oversampled webbpsf
    # nc.detector = detector
    nc.filter = filt
    nc.pixelscale = pixscl
    if date is not None:
        nc.load_wss_opd_by_date(date, plot=False)
    psfraw = nc.calc_psf(oversample=4, fov_arcsec=og_fov, normalize='exit_pupil')
    psf = psfraw['DET_SAMP'].data

    newhdu = fits.PrimaryHDU(psfraw[0].data)
    newhdu.writeto(outname+'_orig.fits', overwrite=True)

    # rotate and handle interpolation internally; keep k = 1 to avoid -ve pixels
    rotated = ndimage.rotate(psf, -angle, reshape=False, order=1, mode='constant', cval=0.0)

    clip = int((og_fov - fov) / 2 / nc.pixelscale)
    rotated = rotated[clip:-clip, clip:-clip]

    # Normalize to correct for missing flux
    # Has to be done encircled! Ensquared were calibated to zero angle...
    w, h = np.shape(rotated)
    Y, X = np.ogrid[:h, :w]
    r = norm_fov / 2. / nc.pixelscale
    center = [w/2., h/2.]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    rotated /= np.sum(rotated[dist_from_center < r])
    rotated *= encircled # to get the missing flux accounted for
    print(f'Final stamp normalization: {rotated.sum()}')

    # and save
    newhdu = fits.PrimaryHDU(rotated)
    newhdu.writeto(outname+'.fits', overwrite=True)


    return rotated

def psf_correction_factor(match_band, psf_dir, apersize=0.32, pixel_scale = 0.03, ):
    conv_psfmodel = fits.open(psf_dir + f'/{match_band}_psf.fits')[0].data # My modelled PSF
    #conv_psfmodel = fits.open('/raid/scratch/data/jwst/PSFs/TomPSFs/PSF_Resample_03_F444W.fits')[0].data # My WebbPSF PSF
    #conv_psfmodel = fits.open('/nvme/scratch/work/tharvey/downloads/MEGASCIENCE_PSFs/f444w_psf_norm.fits')[0].data # UNCOVER/MEGASCIENCE PSF (need 0.04 as pixel size)

    min_corr = 1. / psf_cog(conv_psfmodel, match_band, nearrad=(apersize / 2.), pixel_scale = pixel_scale) # defaults to EE(<R_aper)

    # Convert to AB mag difference

    corr = -2.5 * np.log10(min_corr)

    return corr

if __name__ == '__main__':
    surveys = ['JOF'] #['NEP-1', 'NEP-2', 'NEP-3', 'NEP-4']
    version = 'v11'
    instruments = ['ACS_WFC', 'NIRCam']
    match_band = 'F444W'
    outdir = f'/nvme/scratch/work/tharvey/PSFs/{"+".join(surveys)}/'
    outdir_webbpsf = '/nvme/scratch/work/tharvey/PSFs/webbpsf/morishita_jitter/'
    kernel_dir = f'/nvme/scratch/work/tharvey/PSFs/kernels/{"+".join(surveys)}/'
    maglim = (18.0, 25.0) # Mag limit for stars to stack
    #data = Data.from_pipeline(survey, version = version, instruments = instruments)
    
    #im_paths = data.im_paths
    #bands = data.instrument.band_names
    bands = ['F090W', 'F115W', 'F150W', 'F162M', 'F182M', 'F200W', 'F210M', 'F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F410M', 'F444W']
    folders = [f'/raid/scratch/data/jwst/{survey}/mosaic_1084_wispnathan/' for survey in surveys]
    outdir_matched_images = [f'{folder}psf_matched/' for folder in folders]
    print(folders)
    im_paths = {band:[glob.glob(f'{folder}/*{band.lower()}*.fits')[0] for folder in folders] for band in bands}
    wht_paths = copy(im_paths) # Placeholder
    err_paths = copy(im_paths) # Placeholder
    phot_zp = {band:28.08 for band in bands}

    '''
    # Add HST seperately
    bands.insert(0, 'F606W')
    im_paths['F606W'] = '/raid/scratch/data/hst/NEP-1/ACS_WFC/30mas/aligned_full/ACS_WFC_f606W_NEP-1_drz.fits'
    wht_paths['F606W'] = '/raid/scratch/data/hst/NEP-1/ACS_WFC/30mas/aligned_full/ACS_WFC_f606W_NEP-1_wht.fits'
    err_paths['F606W'] = ''
    
    for band in ['F850LP', 'F814W', 'F775W', 'F606W', 'F435W']:
        bands.insert(0, band)
        im_paths[band] = f'/raid/scratch/data/hst/{field}/ACS_WFC/30mas/aligned_full/ACS_WFC_{band.lower()}_NEP-1_drz.fits'

    hdr = fits.getheader(im_paths['F606W'])
    phot_zp['F606W'] = -2.5 * np.log10(hdr["PHOTFLAM"]) - 21.10 - 5 * np.log10(hdr["PHOTPLAM"]) + 18.6921

    '''
    #kernel_dir_dict = {'aperpy': kernel_dir, 'WebbPSF':'/nvme/scratch/work/tharvey/resolved_sedfitting/psfs/'}
    psf_dir_dict = {'+'.join(surveys):outdir,
    'UNCOVER DR3':'/nvme/scratch/work/tharvey/downloads/MEGASCIENCE_PSFs/', 'WebbPSF Default':'/nvme/scratch/work/tharvey/PSFs/webbpsf/default_jitter/', 'WebbPSF\n$\sigma$=22(SW)/34(LW) mas':'/nvme/scratch/work/tharvey/PSFs/webbpsf/morishita_jitter/'}
    # Rederived 'NEW Webbpsf models have no difference to ours! 
    #'New webbpsf':'/nvme/scratch/work/tharvey/PSFs/webbpsf/'}
    pixelscale = {'+'.join(surveys):0.03, 'WebbPSF Default':0.03, 'UNCOVER DR3':0.04, 'WebbPSF\n$\sigma$=22(SW)/34(LW) mas':0.03}# 'New webbpsf':0.03}

    # Make PSF model and kernels from stacking stars 
   
    #make_psf(bands, im_paths, outdir, kernel_dir, match_band = match_band, phot_zp = phot_zp, maglim=maglim)

    # Generate WebbPSF model for bands - only for comparison!
    for band in bands:
        if int(band[1:-1]) > 240:
            jitter_sigma = 0.034
        else:
            jitter_sigma = 0.022
        print(band, jitter_sigma)
        
        #get_webbpsf(band, field='default', angle=0, fov=4, og_fov=10, pixscl=0.03, date=None, output=outdir_webbpsf, jitter_kernel='gaussian', jitter_sigma=jitter_sigma)

    # Compare EE for different models
    #psf_comparison(bands, psf_dir_dict, match_band=match_band, pixelscale=pixelscale)

    # Convolve images with bands
    #convolve_images(bands, im_paths, wht_paths, err_paths, outdir_matched_images, kernel_dir, match_band, overwrite=True)
    
    # Derive correction factor for apertures from a PSF model 
    #corr = psf_correction_factor('F444W', outdir, apersize=0.32, pixel_scale = 0.03)
    
    #print(corr)