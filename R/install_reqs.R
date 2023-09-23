#' RBPamp installation, requires miniconda
#'
#' @param path_to_package The path to the RegVar R package, check .libPaths if you aren't certain, https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/libPaths
#' @return conda environment- 'RBPamp'- with RBPamp installed, and hg38 fasta installed, ~400MB disc space
#' @examples
#' install_reqs('/Library/Frameworks/R.framework/Versions/4.0/Resources/library/RegVar');
#' @import
#' @export
install_reqs <- function(path_to_package) {
  setwd(path_to_package)
  project_dir<-list.files(pattern = "fai$", recursive = TRUE)[1]
  project_dir<-stringr::str_extract(project_dir[1], '.*extdata.')
  setwd(project_dir)
  #install hg38 fasta
  system('wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz')
  #set up conda environment RBPamp
  setwd('./RBPamp')
  reticulate::conda_create(envname ='RBPamp', environment = 'environment.yml')
  reticulate::conda_install('RBPamp', 'cython')
  path_to_python<-reticulate::conda_list()[[2]][which(reticulate::conda_list()[[1]]=='RBPamp')]
  reticulate::use_python(path_to_python, required = TRUE)
  reticulate::use_condaenv('RBPamp', required=TRUE)
  system('export CC=gcc')
  system('python setup.py build')
  system('pip install .')
}
