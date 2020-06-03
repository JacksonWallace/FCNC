# Plots

Create the following directory:
```
/home/users/$USER/public_html
```

Run the following commands in your public_html directory:
```
setfacl -R -m default:other:r-x .
setfacl -R -m default:mask:r-x .

find . -type d -exec chmod a+rx {} \;
find . -type f -exec chmod a+r {} \;
```

Run `python simplePlots.py`, plots can then be accessed using your web browser:

[http://uaf-10.t2.ucsd.edu/~dspitzba/tW_scattering/](http://uaf-10.t2.ucsd.edu/~dspitzba/tW_scattering/)

Replace your username accordingly. If no plots are visible, run the `chmod` command again.
