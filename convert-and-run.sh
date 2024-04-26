#!/bin/bash
set -e

# Base directory for org files and notebooks
org_base_dir="org"
notebook_base_dir="notebooks"

# Create notebook directory if it doesn't exist
mkdir -p $notebook_base_dir

# Find all .org files in org directory and subdirectories
find $org_base_dir -name "*.org" | while read org_file; do
  # Replace base path from org to notebook directory
  notebook_file="${org_file/$org_base_dir/$notebook_base_dir}"
  notebook_file="${notebook_file%.org}.ipynb"

  # Create subdirectory structure in notebook directory if necessary
  mkdir -p "$(dirname "$notebook_file")"

  # Convert the org file to a notebook file
  pandoc "$org_file" -o "$notebook_file"
done

# # Install nbconvert and other necessary packages for running notebooks
# pip install nbconvert ipykernel matplotlib

# # Run each notebook file within the notebook directory and its subdirectories
# find $notebook_base_dir -name "*.ipynb" -exec jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute '{}' \;
