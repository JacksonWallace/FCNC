# Simulate ttW and tW scattering and survey sensitivity to different EFT operators

We can use inclusive cross sections to make a rough estimate of how sensitive tW scattering is to EFT operators.
We will use the SMEFT model [SMEFTatNLO](http://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO).

## ttW
Simulate ttW with no EWK corrections using
```
import model SMEFTatNLO_U2_2_U3_3_cG_4F_LO_UFO-LO
generate process p p > t t~ w+ NP=2
output ttW_NP2
```
Quit madgraph and navigate to the ttW_NP2 directory.
We can modify `Cards/param_card.dat` and set all the Wilson coefficients to 0 to reproduce the SM ttW process.
All paramaters in block `dim6`, `dim62f`, `dim64f`, `dim64f2l` and `dim64f4l` should be set to 0.
Important: leave `    1 1.000000e+03 # Lambda` unchanged.
Now run `./bin/generate_events` and check the reported cross section.
We can then modify the different Wilson coefficients in `Cards/param_card.dat` and see which ones have an impact on the cross section.
Start by setting cpQ3, cpQM, cpt, ctp, ctZ, ctW to 10 individually and check by how much the cross section changes.

## ttWq
We can repeat the same exercise with the following process
```
import model SMEFTatNLO_U2_2_U3_3_cG_4F_LO_UFO-LO
generate process p p > t t~ w+ QED=3 QCD=1 NP=2
add process p p > t t~ w+ j QED=3 QCD=1 NP=2
output ttW_EWK_NP2
```
We expect to see a larger impact of most of the operators in this process.

## Comparison
Pay attention to the different diagrams of the ttW and ttWq process. What difference do you see?
