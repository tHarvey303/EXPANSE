from ResolvedGalaxy import MockResolvedGalaxy, run_bagpipes_wrapper
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import copy
from joblib import Parallel, delayed
import astropy.units as u
import sys
sys.path.insert(1, 'pipes_scripts/')
from plotpipes import calculate_bins
from pipes_models import delayed_dict, continuity_dict, dpl_dict, lognorm_dict, resolved_dict, create_dicts


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
    # -------------------------------------------------------------------------------------
    
    redshifts=[]
    for redshift_code in list(regions.keys())[::-1]:
        for galaxy_index in range(len(ids[redshift_code])):
            print(f'Doing {redshift_code} galaxy {galaxy_index}')
            try:
                mock_galaxy = MockResolvedGalaxy.init(mock_survey = 'JOF_psfmatched', redshift_code = redshift_code, galaxy_index = galaxy_index, grid_dir = grid_dir, file_path = path, overwrite = overwrite)
                mock_galaxy.pixedfit_processing(gal_region_use = 'detection', overwrite = overwrite) # Maybe seg map should be from detection image?
                mock_galaxy.pixedfit_binning(overwrite = overwrite)
                mock_galaxy.measure_flux_in_bins(overwrite = overwrite)
                run_ids.append(mock_galaxy.galaxy_id)
                redshifts.append(mock_galaxy.redshift)
                del mock_galaxy # Clear memory 

            except AssertionError as e:
                print(e)
                print(f'Failed on {redshift_code} galaxy {galaxy_index}')
                continue
    
    continuity_dicts = create_dicts(continuity_dict, len(run_ids))
    delayed_dicts = create_dicts(delayed_dict, len(run_ids))
    dpl_dicts = create_dicts(dpl_dict, len(run_ids))
    lognorm_dicts = create_dicts(lognorm_dict, len(run_ids))
    resolved_dicts = create_dicts(resolved_dict, len(run_ids))

    # Update continuity_dicts_bins
    for pos, dict in enumerate(continuity_dicts):
        continuity_dicts[pos]['fit_instructions']['continuity']['bin_edges'] = list(calculate_bins(redshift = redshifts[pos], num_bins=6, first_bin=10*u.Myr, second_bin=None, return_flat=True, output_unit='Myr', log_time=False))
    
     
    # Doesn't preserve order otherwise - can't guarantee they will be run in the input order
    for dict in [delayed_dicts, continuity_dicts, dpl_dicts, lognorm_dicts, resolved_dicts]:
        Parallel(n_jobs=n_jobs)(delayed(run_bagpipes_wrapper)(galaxy_id, 
                                                                dic, 
                                                                cutout_size = None, # PLACEHOLDER, not used
                                                                overwrite = False,
                                                                overwrite_internal = True if dic['meta']['run_name'] == 'CNST_SFH_RESOLVED' else False,
                                                                h5_folder=h5_folder) 
                                                for galaxy_id, dic in zip(run_ids, dict))
        
            


            