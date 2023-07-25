
# RegVar

<!-- badges: start -->
<!-- badges: end -->

The goal of RegVar is to characterize 3'UTR variants into their potential regulatory elements.

## Installation
First: You will need a command line to install RegVar; then it can be used in R. With a mac, use the terminal application. With a PC, you will need to download a terminal: https://learn.microsoft.com/en-us/windows/terminal/install.

Then: Make sure you have miniconda, pip, and bedtools installed locally (https://docs.conda.io/en/latest/miniconda.html, https://pip.pypa.io/en/stable/installation/, https://bedtools.readthedocs.io/en/latest/content/installation.html). You can then install the development version of RegVar like so in R:

``` r
#first, install R package: 
devtools::install_github("RomoL2/RegVar") #note, does not install large files
```

Then, in the RegVar directory in command line install the necessary annotation files, like so:
``` r
cd /Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar
mkdir tmp
git clone https://github.com/RomoL2/RegVar ./tmp #clones repository into temporary directory
rm -r extdata #remove directory with symbolic links
mv tmp/inst/extdata ./ #replace with directory with real annotation files
rm -r tmp #ok to override, type "y" #remove temporary directory
```

## Example

Use these functions in R to run RegVar. First use install_reqs (this will install the hg38 fasta, the required conda environment for RBPamp, and RBPamp). Then characterize variants in a standard vcf file with CharacterizeVariants or a single user-input variant with CharacterizeVariants_single_input:

``` r
library(RegVar)
install_reqs('/Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar')
CharacterizeVariants('file.vcf', '~/', '/Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar', '~/')
CharacterizeVariants_single_input("/Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar", "~/")

```

