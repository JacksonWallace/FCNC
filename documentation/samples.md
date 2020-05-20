# Setting up and adding samples

## Config
Check that the config is suiting your needs: `data/config.yaml`

You might want to change the path to the local skims (localSkim) if you want to postprocess nanoAODs.

The version number defines which nanoAOD-tools tag is being checked out, so leave it as is.


## Samples
Add your samples in this file: data/samples.txt.

First column is either the DAS name (for central samples) or the path to the nanoAOD files (privately produced samples).

Second column is the x-sec * BR in pb.

## Getting sample sumw:

Run the script: scripts/getSampleInformation.py

## Submit the jobs:

cd into postProcessing, deactivate your coffea environment, run `source setup.sh`.

`submitter.py` submits all your samples to condor, and resubmits failed jobs every 15mins.

You can run them in a screen session if samples might take longer.
