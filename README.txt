
pylmm is a fast and lightweight linear mixed-model solver for use in genome-wide association studies.

pylmm can be used as a python module to build your own custom programs, or it can used through pylmmGWAS.py to do basic association analysis.  If you want to create your own code, look at example.py for some basic usage patterns.  If you want to run basic GWAS analysis, the basic command below, which uses example data might be a helpful guide.

EXAMPLE COMMAND:

python pylmmGWAS.py -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo

pylmmGWAS.py reads PLINK formmated input file (BED or TPED only).  There is also an option to use "EMMA" formatted files.  We included this in order to make it easier for people using EMMA currently to try pylmm.  The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK or EMMA files as input.  The kinship matrix output is just a plain text file and follows the same format as that used by EMMA, so that you can use pre-computed kinship matrices from EMMA as well, or any other program for that matter.
