#' RBPamp installation (requires miniconda)
#'
#' @param path_to_filename The path to the directory where the filename is located
#' @param path_to_package The path to the R package
#' @return conda environment ('RBPamp') with RBPamp installed, hg38 fasta (~400MB)
#' @examples
#' install_reqs('"~/CharVar");
#' @import
#'
#' @export
installRBPamp <- function(path_to_package) {
  setwd(path_to_package)
  project_dir<-list.files(pattern = "fai$", recursive = TRUE)[1]
  project_dir<-stringr::str_extract(project_dir[1], '.*extdata.')
  setwd(project_dir)
  #check if conda/python/PIP installed
  #install hg38 fasta
  system('wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz')
  #set up conda environment RBPamp
  reticulate::conda_create(envname ='RBPamp', environment = 'environment.yml')
  reticulate::conda_install('RBPamp', 'cython')
  path_to_python<-reticulate::conda_list()[[2]][which(reticulate::conda_list()[[1]]=='RBPamp')]
  reticulate::use_python(path_to_python, required = TRUE)
  reticulate::use_condaenv('RBPamp', required=TRUE)
  system('python setup.py build')
  system('pip install .')
}
