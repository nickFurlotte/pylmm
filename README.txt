
pylmm is a fast and lightweight linear mixed-model solver for use in genome-wide association studies.

pylmm can be used as a python module to build your own custom programs, or it can used through pylmmGWAS.py to do basic association analysis.  If you want to create your own code, look at example.py for some basic usage patterns.  If you want to run basic GWAS analysis, the basic command below, which uses example data might be a helpful guide.

EXAMPLE COMMAND:
python pylmmGWAS.py -v --bfile data/snps.132k.clean.noX --kfile data/snps.132k.clean.noX.pylmm.kin --phenofile data/snps.132k.clean.noX.fake.phenos out.foo

pylmmGWAS.py reads PLINK formmated input file (currently only BED).  The kinship matrix file can be calculated using pylmmKinship.py which also takes PLINK files as input.  However, the kinship matrix file is just a plain text matrix, so any program that calculates such a matrix could be used.
