import copy
import json
import os
import sys
import traceback

import astropy.units as u
import numpy as np
from run_prospector import main, main_parallel

try:
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

    if size > 1:
        print("Running with mpirun/mpiexec detected.")

        MPI.COMM_WORLD.Barrier()
        print(f"Message from process {rank}")
        sys.stdout.flush()
        MPI.COMM_WORLD.Barrier()

except ImportError:
    rank = 0
    size = 1


if __name__ == "__main__":
    try:
        input_json = sys.argv[1]
        parallel_type = sys.argv[2]

        # Can either fit one galaxy on many cores or one galaxy per core.
        if parallel_type == "parallel":
            assert (
                size == 1
            ), "Running this with MPI will also cause problems as we will start a new MPI pool."
            n_jobs = int(sys.argv[3])
            size = n_jobs
            from joblib import Parallel, delayed
        elif parallel_type == "serial":
            pass
        else:
            raise ValueError("parallel_type must be either 'parallel' or 'serial'")

        with open(input_json, "r") as f:
            input_dict = json.load(f)

        run_dicts = {}

        for galaxy in input_dict.keys():
            unique_ids = [f"{galaxy}_{i}" for i in input_dict[galaxy]["ids"]]
            for pos, id in enumerate(unique_ids):
                run_dicts[id] = {}
                run_dicts[id]["OBJID"] = id

                model = copy.deepcopy(input_dict[galaxy]["input_model"])
                model = model[pos] if type(model) is list else model
                run_dicts[id]["input_model"] = model

                meta = copy.deepcopy(input_dict[galaxy]["meta"])
                meta = meta[pos] if type(meta) is list else meta
                run_dicts[id]["meta"] = meta

                filters = copy.deepcopy(input_dict[galaxy]["filters"])
                filters = filters[pos] if type(filters) is list else filters
                run_dicts[id]["filters"] = filters

                phot = copy.deepcopy(input_dict[galaxy]["phot"])
                phot = phot[pos] if type(phot) is list else phot
                phot = np.array(phot)
                assert len(phot.shape) == 2, "Photometry must be a 2D array."
                assert phot.shape[1] == 2, "Photometry must have two columns."
                assert len(phot) == len(
                    filters
                ), "Photometry must have the same number of rows as filters."

                run_dicts[id]["flux"] = phot[:, 0]
                run_dicts[id]["flux_err"] = phot[:, 1]
                run_dicts[id]["min_percentage_err"] = 0  # already applied
                run_dicts[id]["flux_unit"] = u.Jy

                out_dir = f'{input_dict[galaxy]["out_dir"]}/{id}.h5'
                run_dicts[id]["run_path"] = out_dir
                run_dicts[id]["load"] = os.path.exists(out_dir)

                sampling = copy.deepcopy(input_dict[galaxy]["sampling"])
                sampling = sampling[pos] if type(sampling) is list else sampling
                run_dicts[id].update(sampling)  # pass sampling directly.

        if rank == 0:
            print(f"Running from {os.getcwd()} with {size} process{'es' if size > 1 else ''}.")
            print(f"{len(run_dicts)} galax{'y' if len(run_dicts) == 1 else 'ies'} to fit.")

        if parallel_type == "parallel":
            Parallel(n_jobs=n_jobs)(delayed(main)(run_dicts[id]) for id in run_dicts)
        else:
            for id in run_dicts:
                if rank == 0:
                    print(f"Running {id}")
                    sys.stdout.flush()

                main_parallel(run_dicts[id])

        sys.exit(0)
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
