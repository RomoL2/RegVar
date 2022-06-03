#' RBPamp installation (requires miniconda)
#'
#' @param path_to_filename The path to the directory where the filename is located
#' @param path_to_package The path to the R package
#' @return conda environment ('RBPamp') with RBPamp installed
#' @examples
#' CharacterizeVariants('file.vcf', "~/", "~/CharVar/inst/extdata/");
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
  reticulate::use_condaenv('RBPamp', required=TRUE)
  reticulate::py_run_file("setup.py build")
  system('python setup.py build')
  system('pip install .')
}
