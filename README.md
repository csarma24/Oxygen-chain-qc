# Prediction of oxygen drip line using quantum computation

This repository contains the scripts and notebooks used to generate and process
the data from our preprint XXXX.

Packages can be installed from the `requirements.txt`.

Repository structure:
 - The results for the 12-qubit VQE problem can be found in the main directory
 - Resource estimates for the 12-qubit and 5-qubit tapered circuits are in
   `resource-estimation`. Custom circuit ansatze and decompositions can also be
   found in this directory.
 - Notebooks used to construct circuits and submit jobs on the Azure Quantum
   platform are in `hardware-execution`
 - Data from hardware execution and subsequent post-processing and MC
   simulations is in `hardware-data-processing`
