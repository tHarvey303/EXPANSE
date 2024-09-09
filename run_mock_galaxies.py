from ResolvedGalaxy import MockResolvedGalaxy, run_bagpipes_wrapper
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import copy
from joblib import Parallel, delayed

if __name__ == "__main__":

    n_jobs = 8
    overwrite = False
    cosmo = FlatLambdaCDM(H0=70, Om0=0.300)
    grid_name = "bpass-2.2.1-bin_chabrier03-0.1,300.0_cloudy-c17.03"
    grid_dir = "/nvme/scratch/work/tharvey/synthesizer/grids/"
    path = '/nvme/scratch/work/tharvey/synthesizer/flares_flags_balmer_project.hdf5'
    h5_folder = 'galaxies/'

    regions = {'010_z005p000':['00', '00', '01', '10', '18'], '008_z007p000':['00', '02', '09'], '007_z008p000':['21', '17'], '005_z010p000':['15']}
    ids = {'010_z005p000':[12, 96, 1424, 1006, 233], '008_z007p000':[6, 46, 298], '007_z008p000':[111, 16], '005_z010p000':[99]}
    run_ids = []
    num_of_galaxies = np.sum([len(ids[redshift_code]) for redshift_code in list(regions.keys())])

    # ------------------------------------------------------------------------------------- 
    # Setting up two Bagpipes fits - one for the unresolved photometry (delayed SFH) and one for the resolved photometry (constant SFH)

    sfh = {}
    sfh_type = 'delayed'
    sfh["tau"] = (0.01, 15) # `Gyr`
    sfh["massformed"] = (5., 12.)  # Log_10 total stellar mass formed: M_Solar

    sfh["age"] = (0.001, 15) # Gyr
    sfh['age_prior'] = 'uniform'
    sfh['metallicity_prior'] = 'uniform'
    sfh['metallicity'] = (0, 3)

    dust = {}
    dust["type"] = "Calzetti"
    dust["Av"] = (0, 5.0)

    nebular = {}
    nebular["logU"] = (-3.0, -1.0)

    fit_instructions = {"t_bc":0.01,
                    sfh_type:sfh,
                    "nebular":nebular,
                    "dust":dust}

    meta = {'run_name':'photoz_delayed', 'redshift':'self', 'redshift_sigma':'min',
            'min_redshift_sigma':0.5, 'fit_photometry':'TOTAL_BIN',
            'sampler':'multinest'}

    overall_dict = {'meta': meta, 'fit_instructions': fit_instructions}

    dicts = [copy.deepcopy(overall_dict) for i in range(num_of_galaxies)]

    resolved_sfh = {
    'age_max': (0.01, 2.5), # Gyr 
    'age_min': (0, 2.5), # Gyr
    'metallicity': (1e-3, 2.5), # solar
    'massformed': (4, 12), # log mstar/msun
    }

    fit_instructions = {"t_bc":0.01,
                    "constant":resolved_sfh,
                    "nebular":nebular,
                    "dust":dust,  
                    }
    # This means that we are fixing the photo-z to the results from the 'photoz_DPL' run,
    # specifically the 'MAG_APER_TOTAL' photometry
    # We are fitting only the resolved photometry in the 'TOTAL_BIN' bins
    meta = {'run_name':'CNST_SFH_RESOLVED', 'redshift':'photoz_delayed', 'redshift_id':'TOTAL_BIN',
            'fit_photometry':'bin'}

    resolved_dict = {'meta': meta, 'fit_instructions': fit_instructions}
    resolved_dicts = [copy.deepcopy(resolved_dict) for i in range(num_of_galaxies)]
    

    for redshift_code in list(regions.keys())[::-1]:
        for galaxy_index in range(len(ids[redshift_code])):
            print(f'Doing {redshift_code} galaxy {galaxy_index}')
            try:
                mock_galaxy = MockResolvedGalaxy.init(mock_survey = 'JOF_psfmatched', redshift_code = redshift_code, galaxy_index = galaxy_index, grid_dir = grid_dir, file_path = path, overwrite = overwrite)
                mock_galaxy.pixedfit_processing(gal_region_use = 'detection', overwrite = overwrite) # Maybe seg map should be from detection image?
                mock_galaxy.pixedfit_binning(overwrite = overwrite)
                mock_galaxy.measure_flux_in_bins(overwrite = overwrite)
                run_ids.append(mock_galaxy.galaxy_id)
                del mock_galaxy # Clear memory 

            except AssertionError as e:
                print(e)
                print(f'Failed on {redshift_code} galaxy {galaxy_index}')
                continue
    
    
    for run_dicts in [dicts, resolved_dicts]:
        Parallel(n_jobs=n_jobs)(delayed(run_bagpipes_wrapper)(galaxy_id, 
                                                            resolved_dict, 
                                                            cutout_size = None, # PLACEHOLDER, not used
                                                            h5_folder=h5_folder) 
                                            for galaxy_id, resolved_dict in zip(run_ids, run_dicts))
     
        


        