
from distutils.core import setup

#extra={'scripts':['bin/pylmmGWAS.py','pylmmKinship.py']},

setup(
      name='pyLMM',
      version='0.99',
      author = "Nick Furlotte",
      author_email='nick.furlotte@gmail.com',
      url = "https://github.com/nickFurlotte/pylmm",
      description = 'pyLMM is a lightweight linear mixed model solver for use in GWAS.',
      packages=['pylmm'],
      scripts=['bin/pylmmGWAS.py','pylmmKinship.py'],
    )
