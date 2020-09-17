```
export PYTHONPATH=ProjectMetis
```


Start submission in screen session:
```
screen -S sbm
```
Then do
```
source setup.sh
```


Disconnect screen:
ctrl+A, ctrl+D


# NanoAOD tools

We use the following fork for producing babies: [danbarto/nanoAOD-tools/tree/tW_scattering](https://github.com/danbarto/nanoAOD-tools/tree/tW_scattering).
Each submission is done with a unique tag for reproducibility.

Important files to look at are in python/postprocessing/modules/tW_scattering/:
- **GenAnalyzer** loops over the GenParticle collection and writes out the most important generated particles
- **ObjectSelection** has a (basic) object selection, and calculates some event based variables from them
- **lumiWeightProducer** calculates the luminosity weight
- **keep_and_drop** defines what branches to keep

Run the code locally:
- **scripts/run_processor.py** is your place to go within NanoAOD-tools

For local tests just run
```==
python scripts/run_processor.py LUMIWEIGHT
```
where LUMIWEIGHT can be any float number and doesn't really matter for tests (it will only mess up your *weight* branch.

## ToDo
- [ ] Implement years
