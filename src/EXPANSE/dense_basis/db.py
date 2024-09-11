import numpy as np
import glob
import astropy.units as u
from astropy.table import Table
from joblib import Parallel, delayed
import h5py



def get_filter_files(root_file = '/nvme/scratch/work/tharvey/jwst_filters/', 
                    filter_set_name = 'nircam_acs_wfc', 
                    dense_basis_path = '/nvme/scratch/work/tharvey/dense_basis/scripts/filters/filter_curves',
                    instruments={'ACS_WFC':['f435W', 'f606W', 'f814W', 'f775W', 'f850LP'], 
                                'nircam':['f090W', 'f115W', 'f150W', 'f162M', 'f182M', 'f200W', 'f210M', 
                                          'f250M', 'f277W', 'f300M', 'f335M', 'f356W', 'f410M', 'f444W']}, 
                    wav_units = {'ACS_WFC':u.AA, 'nircam':u.um}, 
                    output_wav_unit=u.AA):
    
    filter_files = []
    for instrument in instruments:
        for filter in instruments[instrument]:
            files = glob.glob(f'{root_file}/{instrument}/{filter.upper()}*')
            if len(files) == 0:
                print(f'No files found for {instrument} {filter}')
            elif len(files) > 1:
                print(f'Multiple files found for {instrument} {filter}')
            else:
                tab = Table.read(files[0], format='ascii')
                wav = tab.columns[0] * wav_units[instrument] #first column
                throughput = tab.columns[1] 
                output_tab = Table([wav.to(output_wav_unit), throughput], names=['wav', 'throughput'])
                output_filename = f'{instrument.upper()}_{filter.upper()}.dat'
                output_tab.write(f'{dense_basis_path}/{output_filename}', format='ascii.commented_header', overwrite=True)
                filter_files.append(output_filename)
    # Save list of paths to txt
    path = f'{dense_basis_path}/{filter_set_name}.txt'

    with open(path, 'w') as f:
        for file in filter_files:
            f.write(file+'\n')
    
    return path


def run_db_fit_parallel(obs_sed, obs_err, db_dir, db_atlas_name, atlas_path, fit_mask = [],
                        use_emcee=False, emcee_samples=10_000, plot=False, min_flux_err=0.1):
    '''
    Run a dense_basis fit in parallel
    
    obs_sed: np.array - observed SED in uJy
    obs_err: np.array - observed SED uncertainties in uJy


    '''

    # Set the minimum flux error
    obs_err[obs_err/obs_sed < min_flux_err & obs_sed > 0] = min_flux_err * obs_sed[obs_err/obs_sed < min_flux_err & obs_sed > 0]

    import dense_basis as db
    
    atlas_path = glob.glob(f"{db_dir}/{db_atlas_name}*.dbatlas")[0]
    N_param = int(atlas_path.split("N_param_")[1].split(".dbatlas")[0])
    N_pregrid = int(atlas_path.split("N_pregrid_")[1].split("_N_param")[0])
    
    atlas = db.load_atlas(
        atlas_path, N_pregrid=N_pregrid, N_param=N_param, path=db_dir
    )

    # Need to generate obs_sed, obs_err, and fit_mask based on the input filter files

    if use_emcee:
        sampler = db.run_emceesampler(
            obs_sed, obs_err, atlas, epochs=emcee_samples, plot_posteriors=plot, fit_mask=fit_mask,
        )
    else:
            # pass the atlas and the observed SED + uncertainties into the fitter,
        sedfit = db.SedFit(obs_sed, obs_err, atlas, fit_mask=fit_mask)
        sedfit.evaluate_likelihood()
        sedfit.evaluate_posterior_percentiles()


def make_db_grid(bands, db_dir, filter_set_name = 'JOF',
                fname = 'db_atlas_JOF_',
                N_pregrid = 10000,
                pregrid_path = 'pregrids/',
                N_sfh_priors = 3,
                parameters = {'mass':{'min':5, 'max':12},
                                'Z':{'min':-4, 'max':0.5},
                                'Av':{'min':0, 'max':6},
                                'z':{'min':0, 'max':25}}
                ):
    
    import dense_basis as db

    hst_bands = ['F435W', 'F606W', 'F814W', 'F775W', 'F850LP']

    hst_bands_used = []
    nircam_bands = []

    for band in bands:
        if band in hst_bands:
            hst_bands_used.append(band.replace('F', 'f'))
        else:
            nircam_bands.append(band.replace('F', 'f'))

    path = get_filter_files(dense_basis_path=db_dir,
                            instruments = {'ACS_WFC':hst_bands_used, 'nircam':nircam_bands},
                            filter_set_name=fname)

    priors = db.Priors()

    priors.Nparam = N_sfh_priors

    for param in parameters.keys():
        for key in parameters[param]:
            setattr(priors, f'{param}_{key}', parameters[param][key])

    db.generate_atlas(N_pregrid = N_pregrid,
                    priors = priors,
                    fname = fname, store=True, path=pregrid_path,
                    filter_list = filter_list, filt_dir = filt_dir)

    h5_path = f"{pregrid_path}/{fname}_{N_pregrid}_Nparam_{N_sfh_priors}.dbatlas"

    with h5py.File(h5_path, 'a') as f:
        # add bands as metadata
        f.attrs['bands'] = str(bands)



        
