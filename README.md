
## pylmm - A lightweight linear mixed-model solver

pylmm is a fast and lightweight linear mixed-model (LMM) solver for use in genome-wide association studies (GWAS).

pylmm has a standalone program for running association studies called pylmmGWAS.  It can also be used as a python module to build your own custom programs.  If you want to create your own code, look at example.py for some basic usage patterns.  If you want to run basic GWAS analysis, the command below, which uses example data might be a helpful guide.

### An Example Command:

```
python pylmmGWAS.py -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo
```

The GWAS program pylmmGWAS.py reads PLINK formated input files (BED or TPED only).  There is also an option to use "EMMA" formatted files.  We included this in order to make it easier for people using EMMA currently to try pylmm.  The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.

## Installation 
You will need to have numpy and scipy installed on your current system.
You can install pylmm using pip by doing the following 

```
   pip install git+https://github.com/nickFurlotte/pylmm
```
This should make the module pylmm available as well as the two scripts pylmmGWAS.py and pylmmKinship.py.

You can also clone the repository and do a manual install.
```
   git clone https://github.com/nickFurlotte/pylmm
   python setup.py install
```




