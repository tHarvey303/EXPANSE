import json
import os
import sys
import traceback
import numpy as np
import datetime
import copy

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


def provide_bagpipes_phot(id):
    return np.array(photometry[id])


if __name__ == "__main__":
    try:
        if rank == 0:
            print(f"Running from {os.getcwd()}")
            sys.stdout.flush()
        input_json = sys.argv[1]
        out_subdir = sys.argv[2]

        with open(input_json, "r") as f:
            input_dict = json.load(f)

        (
            ids,
            fit_instructions,
            cat_filt_list,
            redshifts,
            redshift_sigma,
            metas,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        photometry = {}
        for galaxy in input_dict.keys():
            idd = [f"{galaxy}_{i}" for i in input_dict[galaxy]["ids"]]
            ids.extend(idd)  # Make IDs unique
            if type(input_dict[galaxy]["fit_instructions"]) == dict:
                fit_inst = [input_dict[galaxy]["fit_instructions"]] * len(idd)
            elif type(input_dict[galaxy]["fit_instructions"]) == list:
                fit_inst = input_dict[galaxy]["fit_instructions"]
            else:
                raise ValueError(
                    f"fit_instructions must be a dict or a list, not {type(input_dict[galaxy]['fit_instructions'])}"
                )

            if type(input_dict[galaxy]["cat_filt_list"][0]) == str:
                cat_filt_list_t = [copy.deepcopy(input_dict[galaxy]["cat_filt_list"])] * len(idd)
            elif (type(input_dict[galaxy]["cat_filt_list"][0]) == list) & (
                len(input_dict[galaxy]["cat_filt_list"]) == len(idd)
            ):
                cat_filt_list_t = copy.deepcopy(input_dict[galaxy]["cat_filt_list"])

            assert cat_filt_list_t is not None, "cat_filt_list_t cannot be None"
            assert [] not in cat_filt_list_t, "cat_filt_list_t cannot be empty"

            fit_instructions.extend(fit_inst)
            assert len(cat_filt_list_t) == len(
                fit_inst
            ), f"{len(cat_filt_list_t)} != {len(fit_inst)}, {(len(idd))}"
            metas.extend([input_dict[galaxy]["meta"]])
            cat_filt_list.extend(cat_filt_list_t)
            redshifts.extend(input_dict[galaxy]["redshifts"])
            redshift_sigma.extend([input_dict[galaxy]["redshift_sigma"]])
            for pos, id in enumerate(idd):
                photometry[id] = input_dict[galaxy]["phot"][pos]

        assert len(ids) == len(
            fit_instructions
        ), f"Length of IDs ({len(ids)}) and fit_instructions ({len(fit_instructions)}) do not match"

        # Check if use_bpass in meta

        use_bpass = False
        use_bpasses = [meta.get("use_bpass", False) for meta in metas]
        # Check if all are True or all are False
        if all(use_bpasses) or not any(use_bpasses):
            use_bpass = use_bpasses[0]
        else:
            raise ValueError("use_bpass must be the same for all galaxies")

        os.environ["use_bpass"] = str(int(use_bpass))

        # If all cat_filt_list are the same, we can use the same filter list for all galaxies.
        # Otherwise we will need one per bin.
        import bagpipes as pipes

        # Check if all redshift_sigma are None
        if all([rs is None for rs in redshift_sigma]):
            redshift_sigma = None

        # cat_filt_list = cat_filt_list, dtype=object)

        if rank == 0:
            print(f"Running with {size} processes.")
            print(f"{len(ids)} galaxies to fit.")
            print(f"Output directory: {out_subdir}.")
            print(
                len(ids),
                len(fit_instructions),
                len(cat_filt_list),
                len(redshifts),
            )
            print(fit_instructions[0])

        fit_cat = pipes.fit_catalogue(
            ids,
            fit_instructions,
            provide_bagpipes_phot,
            spectrum_exists=False,
            photometry_exists=True,
            run=out_subdir,
            make_plots=False,
            cat_filt_list=cat_filt_list,
            redshifts=redshifts,
            redshift_sigma=redshift_sigma,
            save_pdf_txts=False,
            full_catalogue=True,
            time_calls=False,
            vary_filt_list=True,
        )  # analysis_function=custom_plotting,

        fit_cat.fit(verbose=False, mpi_serial=True)

        # return exit code 0 if successful

        sys.exit(0)
    except Exception as e:
        print(f"Crash on rank {rank} at {datetime.datetime.now()}")
        print(traceback.format_exc())
        print(e)
        sys.exit(1)
