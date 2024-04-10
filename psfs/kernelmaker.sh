#!/bin/bash

# Define a list of filters
filters=("F090W" "F115W" "F150W" "F162M" "F182M" "F200W" "F210M" "F250M" "F277W" "F300M"  "F335M" "F356W" "F360M" "F410M" "F430M")  # Add more filters as needed

# Loop over each filter
for filter in "${filters[@]}"; do
    # Execute the command for each filter
    pypher "webbpsf/${filter}_PSF_003.fits" webbpsf/F444W_PSF_003.fits "kernel_003_${filter}toF444W.fits" 
done
