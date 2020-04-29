# Plots

Run the following commands in your public_html directory:
```
setfacl -R -m default:other:r-x .
setfacl -R -m default:mask:r-x .

find . -type d -exec chmod a+rx {} \;
find . -type f -exec chmod a+r {} \;
```

