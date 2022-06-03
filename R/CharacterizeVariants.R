#' 3'UTR variant characterization
#'
#' @param filename The name of the input vcf file; must be tab delim with 8 columns:
#' chrom, pos, id, ref, alt, qual, filter, info. Chrom column must be either a number (1) or chr# (chr1)
#' @param path_to_filename The path to the directory where the filename is located
#' @param path_to_package The path to the R package
#' #' @return file with variant characterization: processed_filename
#' @examples
#' CharacterizeVariants('file.vcf', "~/", "~/CharVar/inst/extdata/");
#' @import
#' @export
CharacterizeVariants <- function(filename, path_to_filename, path_to_package) {
  #import and format vcf file (format of columns: chrom, chromEnd, name, ref, alt, info)----
  options(scipen = 999)
  setwd(path_to_package)
  project_dir<-list.files(pattern = "fai$", recursive = TRUE)[1]
  project_dir<-stringr::str_extract(project_dir[1], '.*extdata.')
  setwd(project_dir)
  vcf<-data.table::fread(paste(path_to_filename, filename, sep='/'))
  system('gunzip *.gz')
  ##turn vcf into bed and intersect with 3'UTR (bedtools)
  names(vcf)<-c('chrom', 'chromEnd', 'name', 'ref', 'alt', 'qual', 'filter', 'info')
  vcf[, c('qual', 'filter') := NULL]
  #format chr column
  if(grepl('chr', vcf$chrom[1], fixed=TRUE)==FALSE) {
    vcf[, chrom := paste('chr', chrom, sep='')]
  }
  #keep only SNVs
  vcf<-vcf[nchar(ref)==1 & nchar(alt)==1,]
  vcf[, chromStart := chromEnd-1]
  vcf[, name := paste(ref, alt, info, sep='__')]
  vcf<-vcf[, c('chrom', 'chromStart', 'chromEnd', 'name')]
  data.table::fwrite(vcf, 'vcf.bed', sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE)

  #intersect with UTR (3pseq and ucsc)
  system('bedtools intersect -a vcf.bed -b all_APA_peak_coords_hg38.bed -wa -wb > vcf_UTR.bed')
  vcf_UTR<-data.table::fread('vcf_UTR.bed')
  #something here to report if this file is empty, no UTR variants
  names(vcf_UTR)<-c('chrom', 'chromStart', 'chromEnd', 'info',
                    'chr2', 'isoStart', 'isoStop', 'gene', 'score', 'strand',
                    'number_isos', 'iso_loc', 'UTRstart', 'UTRstop')
  vcf_UTR[, c('chr2', 'score') := NULL]

  #intersect with eclip----
  print('incorporating eCLIP')
  vcf_UTR[, tmp_key:=seq(1:nrow(vcf_UTR))]
  data.table::fwrite(vcf_UTR[, c('chrom', 'chromStart', 'chromEnd', 'tmp_key')], 'vcf_UTR_tmp.bed', col.names=FALSE, row.names=FALSE, quote=FALSE, sep='\t')
  #intersect with hepg2 and k562 total eclip
  system("bedtools intersect -a vcf_UTR_tmp.bed -b eCLIP_peaks.GRCh38.IDR_pass.main_chr.HepG2.full_name.bed -wa -wb > vcf_UTR_tmp_hepg2.txt")
  system("bedtools intersect -a vcf_UTR_tmp.bed -b eCLIP_peaks.GRCh38.IDR_pass.main_chr.K562.full_name.bed -wa -wb > vcf_UTR_tmp_k562.txt")
  vcf_UTR_hepg2<-data.table::fread('vcf_UTR_tmp_hepg2.txt')
  vcf_UTR_hepg2<-vcf_UTR_hepg2[, c('V4', 'V8')]
  names(vcf_UTR_hepg2)<-c('tmp_key', 'eclip_id')
  vcf_UTR_hepg2[, tot_eclip_RBP:=data.table::tstrsplit(eclip_id, '_')[1]]
  vcf_UTR_hepg2[, tot_eclip_RBP:=paste(tot_eclip_RBP, 'HepG2', sep='_')]
  vcf_UTR_k562<-data.table::fread('vcf_UTR_tmp_k562.txt')
  vcf_UTR_k562<-vcf_UTR_k562[, c('V4', 'V8')]
  names(vcf_UTR_k562)<-c('tmp_key', 'eclip_id')
  vcf_UTR_k562[, tot_eclip_RBP:=data.table::tstrsplit(eclip_id, '_')[1]]
  vcf_UTR_k562[, tot_eclip_RBP:=paste(tot_eclip_RBP, 'K562', sep='_')]
  #add info to vcf_UTR
  vcf_UTR[, eclip_tot:='']
  for (n in 1:nrow(vcf_UTR)) {
    matches_k562<-vcf_UTR_k562[tmp_key==vcf_UTR$tmp_key[n]]$tot_eclip_RBP
    matches_hepg2<-vcf_UTR_hepg2[tmp_key==vcf_UTR$tmp_key[n]]$tot_eclip_RBP
    if (length(matches_k562)>0 & length(matches_hepg2)>0) {
      all_matches<-paste(paste(matches_k562, collapse='__'), paste(matches_hepg2, collapse='__'), sep='__')
      vcf_UTR[n, eclip_tot:=all_matches]
    } else if (length(matches_k562)>0 & length(matches_hepg2)==0) {
      all_matches<-paste(matches_k562, collapse='__')
      vcf_UTR[n, eclip_tot:=all_matches]
    } else if (length(matches_k562)==0 & length(matches_hepg2)>0) {
      all_matches<-paste(matches_hepg2, collapse='__')
      vcf_UTR[n, eclip_tot:=all_matches]
    }
  }
  vcf_UTR[, tmp_key:=NULL]
  vcf_UTR[, ref := data.table::tstrsplit(info, '__')[[1]]]
  vcf_UTR[, alt := data.table::tstrsplit(info, '__')[[2]]]
  rm(vcf_UTR_hepg2, vcf_UTR_k562)

  #intersect with (nearby +/-5bp) eQTLs----
  print('incorporating eQTLS')
  vcf_UTR[, tmp_key := seq(1:nrow(vcf_UTR))]

  #write bed file for intersect, then match back to original
  data.table::fwrite(vcf_UTR[, c('chrom', 'chromStart', 'chromEnd', 'tmp_key')], 'vcf_UTR_tmp.bed', col.names=FALSE, row.names=FALSE, quote=FALSE, sep='\t')
  system('bedtools intersect -a vcf_UTR_tmp.bed -b GTEx_v8_finemapping_DAPG_UTR_for_script.bed -wa -wb > vcf_UTR_eqtls.bed')
  tmp<-data.table::fread('vcf_UTR_eqtls.bed')
  names(tmp)<-c('chrom', 'chromStart', 'chromEnd', 'tmp_key', 'chr2', 'window_start', 'window_stop', 'eqtl_info')
  tmp<-unique(tmp[, .(tmp_key, eqtl_info)]) #eqtls have 3pseq coords, 10bp window around eqtl

  #add to vcf file
  data.table::setkey(tmp, tmp_key)
  data.table::setkey(vcf_UTR, tmp_key)
  vcf_UTR<-vcf_UTR[tmp]

  #eqtl info column: chromStart_chromEnd_ref_alt_signalid@tissue_name= PIP[SPIP:size_of_cluster]
  # more specifically:
  # + signalid: ID of a signal cluster. It consists of the name of the gene and the signal index separated by ":". e.g., ENSG00000238009:3 indicates the signal is the third signal from gene ENSG00000238009
  # + tissue_name: name of the tissue where the SNP is investigated
  # + PIP: SNP posterior inclusion probability. Higher PIP value indicates the SNP is more likely to be the casual eQTL.
  # + SPIP: signal-level posterior_inclusion probability (sum of the PIPs from all members of the signal cluster)
  # + size_of_cluster: number of SNPs in the signal cluster. These member SNPs are in LD, all represent the same underlying association signal

  #intersect with (nearby +/-5bp) GWAS----
  print('incorporating GWAS')
  #write bed file for intersect, then match back to original
  vcf_UTR[, tmp_key := seq(1:nrow(vcf_UTR))]
  data.table::fwrite(vcf_UTR[, c('chrom', 'chromStart', 'chromEnd', 'tmp_key')], 'vcf_UTR_tmp.bed', col.names=FALSE, row.names=FALSE, quote=FALSE, sep='\t')
  system('bedtools intersect -a vcf_UTR_tmp.bed -b GWAS_UTR_for_script.bed -wa -wb > vcf_UTR_gwas.bed')
  tmp<-data.table::fread('vcf_UTR_gwas.bed')
  names(tmp)<-c('chrom', 'chromStart', 'chromEnd', 'tmp_key', 'chr2', 'window_start', 'window_stop', 'gwas_info')
  tmp<-unique(tmp[, .(tmp_key, gwas_info)])

  #add to vcf file
  data.table::setkey(tmp, tmp_key)
  data.table::setkey(vcf_UTR, tmp_key)
  vcf_UTR<-vcf_UTR[tmp]
  #gwas info column: chromStart, chromEnd, rsID, minor_allele, ref, alt, fine_map, pheno, maf, effect_size, pip

  #intersect with microRNAs from TargetScan----
  print('incorporating microRNAs')
  vcf_UTR[, tmp_key := seq(1:nrow(vcf_UTR))]
  data.table::fwrite(vcf_UTR[, c('chrom', 'chromStart', 'chromEnd', 'tmp_key')], 'vcf_UTR_tmp.bed', col.names=FALSE, row.names=FALSE, quote=FALSE, sep='\t')
  system('bedtools intersect -a vcf_UTR_tmp.bed -b hg38_miR_predictions_final.bed -wa -wb > vcf_UTR_miRs.bed')
  tmp<-data.table::fread('vcf_UTR_miRs.bed')
  names(tmp)<-c('chrom', 'chromStart', 'chromEnd', 'tmp_key', 'chr2', 'miRstart', 'miRstop', 'mergekey')
  tmp<-unique(tmp[, .(tmp_key, mergekey)])

  #add miR mergekey (unique key to match miR bed locations to full miR info file) to vcf file
  data.table::setkey(tmp, tmp_key)
  data.table::setkey(vcf_UTR, tmp_key)
  vcf_UTR<-vcf_UTR[tmp]

  #merge with rest of miR info using mergekey
  miR_predictions[, cons := substr(cons, 1, nchar(cons)-4)]
  miR_predictions[, i.cons := substr(i.cons, 1, nchar(i.cons)-4)]
  miR_predictions[, i.cons := data.table::tstrsplit(i.cons, '_')[[2]]]
  miR_predictions[, miR_info := paste(miR_family, seed_match, Pct, strand, context_pile, cons, i.cons, sep='__')]
  miR_predictions<-miR_predictions[, .(mergekey, miR_info)]
  data.table::setkey(miR_predictions, mergekey)
  data.table::setkey(vcf_UTR, mergekey)
  vcf_UTR<-miR_predictions[vcf_UTR]
  vcf_UTR[, mergekey := NULL]
  #miR_info column: miR_family__seed_match__Pct__strand__context_pile__familycons__sitecons
  rm(miR_predictions)

  #RBP affinity scoring with RBPamp (need installed locally first)----
  print('incorporating RBP motifs')
  ##prep variants ----
  vcf_UTR[, RBPamp_info := paste(ref, alt, sep="_")]
  names_BED_std <- c("chrom", "chromStart", "chromEnd", "name", "score", "strand")
  data.table::fwrite(unique(vcf_UTR[, c('chrom', 'chromStart', 'chromEnd', 'RBPamp_info', 'gene', 'strand')]), 'vcf_UTR_tmp.bed', col.names=FALSE, row.names=FALSE, quote=FALSE, sep='\t')
  system('sort -V vcf_UTR_tmp.bed > nat_sorted_vcf_UTR_tmp.bed')

  # merge BY STRAND to get final file (use d -1 to avoid merging bookended adjacent features)
  system("bedtools merge -i  nat_sorted_vcf_UTR_tmp.bed -s -d -1 -c 4,5,6 -o distinct,distinct,distinct > merged_by_strand_nat_sorted_vcf_UTR_tmp.bed")
  # note: this will also merge multi-allelic sites (with more than one alt), so need to expand them below

  # read intersection in
  intersection <- data.table::fread("merged_by_strand_nat_sorted_vcf_UTR_tmp.bed", header = FALSE)
  names(intersection) <- names_BED_std
  # assign allele numbers
  intersection[, allele_count := stringr::str_count(name, stringr::fixed(",")) + 2]
  if(max(intersection$allele_count)>4) {
    stop('more than four alleles at chromosome position')
  }

  ## expand multi-allelic sites if present (separate and expand so there is one line per allele)
  # isolate single allele variants from original datatable
  singles <- intersection[allele_count == 2, ..names_BED_std]
  # expand triple and quadruple alleles
  if (max(intersection$allele_count)==3) {
    triples <- intersection[allele_count == 3, ]
    triples[, c("name_1", "name_2") := data.table::tstrsplit(name, ",", fixed = TRUE)]
    triples_expanded <- rbind(triples[, .(chrom, chromStart, chromEnd, name = name_1, score, strand)],
                              triples[, .(chrom, chromStart, chromEnd, name = name_2, score, strand)])
    data.table::setkeyv(triples_expanded, c("chrom", "chromStart"))
    intersection <- rbind(singles, triples_expanded)
  } else if (max(intersection$allele_count)==4) {
    quads <- intersection[allele_count == 4, ]
    quads[, c("name_1", "name_2", "name_3") := data.table::tstrsplit(name, ",", fixed = TRUE)]
    quads_expanded <- rbind(quads[, .(chrom, chromStart, chromEnd, name = name_1, score, strand)],
                            quads[, .(chrom, chromStart, chromEnd, name = name_2, score, strand)],
                            quads[, .(chrom, chromStart, chromEnd, name = name_3, score, strand)])
    data.table::setkeyv(quads_expanded, c("chrom", "chromStart"))
    intersection <- rbind(singles, triples_expanded, quads_expanded)
  } else {
    intersection <- singles
  }
  variants<-intersection

  suppressWarnings(
    rm(singles, triples_expanded, quads_expanded, triples, quads, intersection)
  )

  ##strand conversion ----
  variants[, 'ref_plus' := data.table::tstrsplit(variants$name, "_", fixed = TRUE)[1]]
  variants[, 'alt_plus' := data.table::tstrsplit(variants$name, "_", fixed = TRUE)[2]]

  # correct sequence based on strand
  variants[strand == "+", ref_strand := ref_plus]
  variants[strand == "+", alt_strand := alt_plus]
  variants[strand == "-", ref_strand := stringi::stri_reverse(chartr("ACGT", "TGCA", ref_plus))]
  variants[strand == "-", alt_strand := stringi::stri_reverse(chartr("ACGT", "TGCA", alt_plus))]

  # re-package name w/ !stranded! ref/alt
  variants[, name := paste(name, ref_strand, alt_strand, sep = "_")] #first two are original vcf, second are w/ strand conversion
  # remove un-needed columns
  variants <- variants[, ..names_BED_std]

  ##add sequence contexts and write files ----
  # set dist to add to either side of variant (for full context to determine if focal or non focal)
  # use 43 bases (mean of eCLIP peaks = 66, plus/minus 10, divided by 2)
  k_to_add_full <- 43
  # update chromStart and chromEnd (gives a total of 87, but allows even extension on both sides which is important for subsequent steps)
  variants[, chromStart_k := as.integer(chromStart - k_to_add_full)]
  variants[, chromEnd_k := as.integer(chromEnd + k_to_add_full)]

  # add original chromStart and chromEnd for the variant to the name column
  variants[, name := paste(chromStart, chromEnd, name, sep = "_")]

  # write to file
  data.table::fwrite(variants[, .(chrom, chromStart_k, chromEnd_k, name, score, strand)], "nat_sorted_vcf_UTR_tmp_intersected_expanded_RNA_strand_PLUS_k_for_seq2.bed", sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE)
  # fetch surrounding sequences
  system("bedtools getfasta -bedOut -s -fi hg38.fa -bed nat_sorted_vcf_UTR_tmp_intersected_expanded_RNA_strand_PLUS_k_for_seq2.bed > nat_sorted_vcf_UTR_tmp_intersected_expanded_RNA_strand_PLUS_k_for_seq2.txt")

  # read back in
  variants <- data.table::fread("nat_sorted_vcf_UTR_tmp_intersected_expanded_RNA_strand_PLUS_k_for_seq2.txt", header = FALSE)
  names(variants) <- c(names_BED_std, "seq")
  variants[, c("chromStart", "chromEnd") := NULL]

  # hg38.fa is a soft-masked file. bases covered by repeat masker are shown in lower case
  # convert to upper-case
  variants[, seq := toupper(seq)]

  # remove any entries with N base calls
  variants <- variants[!(grepl("N", variants[, seq], fixed = TRUE)), ]

  # extract reference base from seq
  variants[, ref_base := substr(seq, k_to_add_full + 1, k_to_add_full + 1)]

  #NTs in name: first two are original vcf, second are w/ strand conversion
  variants[, c("chromStart", "chromEnd", "ref", "alt", "ref_conv", "alt_conv") := data.table::tstrsplit(name, "_", fixed = TRUE)]
  if(sum(variants[, ref_base == ref_conv]) != nrow(variants)) {
    warning('one ore more hg38 bases do not match your vcf bases')
  } # make sure base from hg38 fasta matches strand-converted base from variant call data
  #restore name column
  variants[, name:= paste(chromStart, chromEnd, ref, alt, sep='_')]

  # set desired sequence window for calculating affinity of RBP binding sites overlapping variant (use k = 11 for max, corresponds to max footprint size used by RBPamp, and k = 10 for some RBPs)
  k_to_add_var <- 11
  variants[, seq_ref_11 := substr(seq, (k_to_add_full + 1) - (k_to_add_var -1), (k_to_add_full + 1) + (k_to_add_var -1))]
  k_to_add_var <- 10
  variants[, seq_ref_10 := substr(seq, (k_to_add_full + 1) - (k_to_add_var -1), (k_to_add_full + 1) + (k_to_add_var -1))]

  # add alt seq for k = 11 and k = 10
  variants[, seq_alt_11 := paste(substr(seq_ref_11, 1, k_to_add_var -1), ref_conv, substr(seq_ref_11, k_to_add_var + 1, k_to_add_var + k_to_add_var - 1), sep = "")]
  variants[, seq_alt_10 := paste(substr(seq_ref_10, 1, k_to_add_var -1), ref_conv, substr(seq_ref_10, k_to_add_var + 1, k_to_add_var + k_to_add_var - 1), sep = "")]

  # write to files
  # full seqs
  temp_seq <- data.table::data.table(seq = unique(variants[, seq]))
  data.table::setkey(temp_seq, seq)
  data.table::fwrite(temp_seq, "vars.full_seq.txt", sep = "\t", quote = FALSE, col.names = TRUE, row.names = FALSE)
  rm(temp_seq)
  # k = 11 seqs
  temp_seq <- data.table::data.table(seq = unique(variants[, c(seq_ref_11, seq_alt_11)])) # use "c" not "."
  data.table::setkey(temp_seq, seq)
  data.table::fwrite(temp_seq, "vars.k_11_seq.txt", sep = "\t", quote = FALSE, col.names = TRUE, row.names = FALSE)
  rm(temp_seq)
  # k = 10 seqs
  temp_seq <- data.table::data.table(seq = unique(variants[, c(seq_ref_10, seq_alt_10)])) # use "c" not "."
  data.table::setkey(temp_seq, seq)
  data.table::fwrite(temp_seq, "vars.k_10_seq.txt", sep = "\t", quote = FALSE, col.names = TRUE, row.names = FALSE)
  rm(temp_seq)

  ##fetch affinities----
  path_to_python<-reticulate::conda_list()[[2]][which(reticulate::conda_list()[[1]]=='RBPamp')]
  reticulate::use_python(path_to_python, required = TRUE)
  reticulate::use_condaenv('RBPamp', required=TRUE)
  reticulate::source_python("RBPamp_aff_local.py") #need RBP motif files in ./motifs2

  #read back in
  full_affs <- data.table::fread("vars.full_seq.RBPamp_affs.motifs.tsv", header = FALSE)
  names(full_affs) <- c("seq", "RBP", "k_RBP", "aff")
  k_11_affs <- data.table::fread("vars.k_11_seq.RBPamp_affs.motifs.tsv", header = FALSE)
  names(k_11_affs) <- c("seq", "RBP", "aff")
  k_10_affs <- data.table::fread("vars.k_10_seq.RBPamp_affs.motifs.tsv", header = FALSE)
  names(k_10_affs) <- c("seq", "RBP", "aff")
  names(full_affs) <- c("seq", "RBP", "k_RBP", "aff_full")
  data.table::setkey(full_affs, seq)
  data.table::setkey(variants, seq)
  variants <- full_affs[variants, allow.cartesian = TRUE] # allows expansion of variants so there's entries for each RBP

  # split into k = 11 RBPs and k = 10 RBPs
  vars_k_11 <- variants[k_RBP == 11, ]
  vars_k_11[, c("seq_ref_10", "seq_alt_10") := NULL]
  data.table::setnames(vars_k_11, c("seq_ref_11", "seq_alt_11"), c("seq_ref", "seq_alt"))
  vars_k_10 <- variants[k_RBP == 10, ]
  vars_k_10[, c("seq_ref_11", "seq_alt_11") := NULL]
  data.table::setnames(vars_k_10, c("seq_ref_10", "seq_alt_10"), c("seq_ref", "seq_alt"))

  # add affs to k = 11 entries
  names(k_11_affs) <- c("seq_ref", "RBP", "aff_ref")
  data.table::setkey(k_11_affs, seq_ref, RBP)
  data.table::setkey(vars_k_11, seq_ref, RBP)
  vars_k_11 <- k_11_affs[vars_k_11, nomatch = 0]
  names(k_11_affs) <- c("seq_alt", "RBP", "aff_alt")
  data.table::setkey(k_11_affs, seq_alt, RBP)
  data.table::setkey(vars_k_11, seq_alt, RBP)
  vars_k_11 <- k_11_affs[vars_k_11, nomatch = 0]

  # add affs to k = 10 entries
  names(k_10_affs) <- c("seq_ref", "RBP", "aff_ref")
  data.table::setkey(k_10_affs, seq_ref, RBP)
  data.table::setkey(vars_k_10, seq_ref, RBP)
  vars_k_10 <- k_10_affs[vars_k_10, nomatch = 0]
  names(k_10_affs) <- c("seq_alt", "RBP", "aff_alt")
  data.table::setkey(k_10_affs, seq_alt, RBP)
  data.table::setkey(vars_k_10, seq_alt, RBP)
  vars_k_10 <- k_10_affs[vars_k_10, nomatch = 0]

  # combine k = 11 and k = 11 entries back together
  rm(variants)
  variants <- rbind(vars_k_11, vars_k_10)
  rm(vars_k_10, vars_k_11)

  #strength threshold
  strong_prop <- 0.33 ##make customizable??
  lost_prop <- 0.33
  pres_prop <- 0.67

  variants[aff_ref > strong_prop, motif_strong := 1]
  variants[aff_ref <= strong_prop, motif_strong := 0]
  #lost
  variants[aff_ref > strong_prop & (aff_alt / aff_ref) < lost_prop, cat := "lost"]
  #preserved
  variants[aff_ref > strong_prop & (aff_alt / aff_ref) > pres_prop, cat := "preserved"]

  ##collapse variants in more than one RBP motif----
  variants<-variants[!is.na(cat)] #only keep variants in strong motifs
  variants[, base_id:=paste(chrom, chromStart, chromEnd, strand, sep='_')]
  variants[, var_id:=paste(base_id, ref, alt, sep='_')]
  unique_vars<-unique(variants$var_id)
  compressed_variants<-data.table::data.table()
  variants<-unique(variants[, c('var_id', 'RBP', 'cat')])
  for (n in 1:length(unique_vars)) {
    matches<-variants[var_id==unique_vars[n],]
    if (nrow(matches)>0) {
      RBPs<-paste(matches$RBP, collapse = "_")
      cat<-paste(matches$cat, collapse='__')
      row<-cbind(matches$var_id[1], RBPs, cat)
      compressed_variants<-rbind(compressed_variants, row)
    }
  }
  names(compressed_variants)<-c('var_id', 'motif_RBPs', 'motif_cat')

  ##merge RBPamp info back to variants  ----
  vcf_UTR[, RBPamp_info := NULL]
  vcf_UTR[, var_id := paste(chrom, chromStart, chromEnd, strand, ref, alt, sep='_')]
  data.table::setkey(vcf_UTR, var_id)
  data.table::setkey(compressed_variants, var_id)
  vcf_UTR<-compressed_variants[vcf_UTR]

  #CADD (only gnomad vars, too much disc space to run whole thing)----
  print('fetching CADD scores for gnomAD variants')
  names(cadd_gd)<-c('chrom', 'chromStart', 'chromEnd', 'cadd_var_info')
  cadd_gd[, ref := data.table::tstrsplit(cadd_var_info, '_')[[1]]]
  cadd_gd[, alt := data.table::tstrsplit(cadd_var_info, '_')[[2]]]
  cadd_gd[, merge_id := paste(chrom, chromStart, chromEnd, ref, alt, sep="_")]
  cadd_gd<-cadd_gd[, .(cadd_var_info, merge_id)]
  vcf_UTR[, merge_id := paste(chrom, chromStart, chromEnd, ref, alt, sep="_")]
  data.table::setkey(vcf_UTR, merge_id)
  data.table::setkey(cadd_gd, merge_id)
  vcf_UTR<-cadd_gd[vcf_UTR]
  vcf_UTR[, merge_id := NULL]
  #cadd_var_info: ref_alt_RawScore_PHRED
  rm(cadd_gd)

  #nearby (+/-5bp) ClinVar----
  print('incorporating ClinVar')
  vcf_UTR[, tmp_key := seq(1:nrow(vcf_UTR))]
  data.table::fwrite(vcf_UTR[, c('chrom', 'chromStart', 'chromEnd', 'tmp_key')], 'vcf_UTR_tmp.bed', col.names=FALSE, row.names=FALSE, quote=FALSE, sep='\t')
  system('bedtools intersect -a vcf_UTR_tmp.bed -b clinvar_for_script.bed -wa -wb > vcf_UTR_clinvar.bed')
  tmp<-data.table::fread('vcf_UTR_clinvar.bed')
  names(tmp)<-c('chrom', 'chromStart', 'chromEnd', 'tmp_key', 'chr2', 'window_start', 'window_stop', 'clinvar_info')
  tmp<-unique(tmp[, .(tmp_key, clinvar_info)])

  #add to vcf file
  data.table::setkey(tmp, tmp_key)
  data.table::setkey(vcf_UTR, tmp_key)
  vcf_UTR<-vcf_UTR[tmp]
  vcf_UTR[, tmp_key := NULL]
  #clinvar info column:
  ##INFO=<ID=CLNDNINCL,Number=.,Type=String,Description="For included Variant : ClinVar's preferred disease name for the concept specified by disease identifiers in CLNDISDB">
  ##INFO=<ID=CLNDISDB,Number=.,Type=String,Description="Tag-value pairs of disease database name and identifier, e.g. OMIM:NNNNNN">
  ##INFO=<ID=CLNDISDBINCL,Number=.,Type=String,Description="For included Variant: Tag-value pairs of disease database name and identifier, e.g. OMIM:NNNNNN">
  ##INFO=<ID=CLNHGVS,Number=.,Type=String,Description="Top-level (primary assembly, alt, or patch) HGVS expression.">
  ##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description="ClinVar review status for the Variation ID">
  ##INFO=<ID=CLNSIG,Number=.,Type=String,Description="Clinical significance for this single variant">
  ##INFO=<ID=CLNSIGCONF,Number=.,Type=String,Description="Conflicting clinical significance for this single variant">
  ##INFO=<ID=CLNSIGINCL,Number=.,Type=String,Description="Clinical significance for a haplotype or genotype that includes this variant. Reported as pairs of VariationID:clinical significance.">
  ##INFO=<ID=CLNVC,Number=1,Type=String,Description="Variant type">
  ##INFO=<ID=CLNVCSO,Number=1,Type=String,Description="Sequence Ontology id for variant type">
  ##INFO=<ID=CLNVI,Number=.,Type=String,Description="the variant's clinical sources reported as tag-value pairs of database and variant identifier">
  ##INFO=<ID=DBVARID,Number=.,Type=String,Description="nsv accessions from dbVar for the variant">
  ##INFO=<ID=GENEINFO,Number=1,Type=String,Description="Gene(s) for the variant reported as gene symbol:gene id. The gene symbol and id are delimited by a colon (:) and each pair is delimited by a vertical bar (|)">
  ##INFO=<ID=MC,Number=.,Type=String,Description="comma separated list of molecular consequence in the form of Sequence Ontology ID|molecular_consequence">
  ##INFO=<ID=ORIGIN,Number=.,Type=String,Description="Allele origin. One or more of the following values may be added: 0 - unknown; 1 - germline; 2 - somatic; 4 - inherited; 8 - paternal; 16 - maternal; 32 - de-novo; 64 - biparental; 128 - uniparental; 256 - not-tested; 512 - tested-inconclusive; 1073741824 - other">
  ##INFO=<ID=RS,Number=.,Type=String,Description="dbSNP ID (i.e. rs number)">
  ##INFO=<ID=SSR,Number=1,Type=Integer,Description="Variant Suspect Reason Codes. One or more of the following values may be added: 0 - unspecified, 1 - Paralog, 2 - byEST, 4 - oldAlign, 8 - Para_EST, 16 - 1kg_failed, 1024 - other">

  #APA info----
  print('incorporating APA info')
  #isoform region
  vcf_UTR$dvp<-vcf_UTR$iso_loc/vcf_UTR$number_isos
  vcf_UTR[number_isos>1 & dvp<=0.5, iso_region:='common']
  vcf_UTR[number_isos==1, iso_region:='single']
  vcf_UTR[number_isos>1 & dvp>0.5 & dvp<1, iso_region:='partially_shared']
  vcf_UTR[number_isos>1 & dvp==1, iso_region:='unique']
  vcf_UTR[, dvp := NULL]

  #distance to PAS and stop codon
  vcf_UTR[,PAS_1:=abs(chromEnd-isoStart)]
  vcf_UTR[,PAS_2:=abs(chromEnd-isoStop)]
  vcf_UTR[PAS_1<PAS_2, PAS_1:=PAS_1]
  vcf_UTR[PAS_1>=PAS_2, PAS_1:=PAS_2]
  vcf_UTR[,PAS_2:=NULL]
  vcf_UTR[, stop_d:=abs(chromStart-UTRstart)]

  #conservation info ----
  print('incorporating conservation scores')
  cons_info<-data.table::fread('LR_all_APA_peak_coords_hg38_by_base.phylop_100.phylop_17.phastcons_100.phastcons_17.txt')
  cons_info[, base_id := paste(chrom, chromStart, chromEnd, sep='_')]
  cons_info<-cons_info[, .(base_id, phastcons_100, phylop_100)]
  data.table::setkey(cons_info, base_id)
  vcf_UTR[, base_id := paste(chrom, chromStart, chromEnd, sep='_')]
  data.table::setkey(vcf_UTR, base_id)
  vcf_UTR<-cons_info[vcf_UTR]
  vcf_UTR[, base_id := NULL]
  rm(cons_info)

  #predicted eQTL or GWAS (PIP>0.5) based on GLMs----
  print('predicting GWAS or eQTLs')
  vcf_UTR[, in_miR:=ifelse(grepl('consFam', miR_info, perl=TRUE) |
                             grepl('broadConsFam', miR_info, perl=TRUE), 1, 0)]
  vcf_UTR[, PAS := ifelse(PAS_1<=50, 1, 0)]
  vcf_UTR[, cons := ifelse(phastcons_100>0.5, 1, 0)]
  vcf_UTR[, region := ifelse(iso_region=='common' | iso_region=='single', 1, 0)]
  vcf_UTR[, in_eclip := ifelse(eclip_tot!='', 1, 0)]

  #apply model to user data
  vcf_UTR[, pr_eqtl := predict(eqtl_model, vcf_UTR, type='response')]
  vcf_UTR[, pred_eqtl := ifelse(pr_eqtl>0.1, 1, 0)] #best sensitivity/specificity log-odds threshold
  vcf_UTR[, pr_gwas := predict(gwas_model, vcf_UTR, type='response')]
  vcf_UTR[, pred_gwas := ifelse(pr_gwas>0.0075, 1, 0)] #best sensitivity/specificity log-odds threshold

  #format and write outputs and clean up disc (remove intermediates)----
  print('formatting output')
  #remove intermediates
  system('rm vcf.bed vcf_UTR.bed vcf_UTR_tmp.bed vcf_UTR_tmp_hepg2.txt vcf_UTR_tmp_k562.txt vcf_UTR_eqtls.bed vcf_UTR_gwas.bed vcf_UTR_miRs.bed nat_sorted_vcf_UTR_tmp.bed merged_by_strand_nat_sorted_vcf_UTR_tmp.bed nat_sorted_vcf_UTR_tmp_intersected_expanded_RNA_strand_PLUS_k_for_seq2.bed nat_sorted_vcf_UTR_tmp_intersected_expanded_RNA_strand_PLUS_k_for_seq2.txt vars.full_seq.txt vars.k_11_seq.txt vars.k_10_seq.txt vcf_UTR_clinvar.bed vars.k_11_seq.RBPamp_affs.motifs.tsv vars.full_seq.RBPamp_affs.motifs.tsv vars.k_10_seq.RBPamp_affs.motifs.tsv')
  #format output
  vcf_UTR[, var_id := paste(chrom, chromEnd, ref, alt, sep='_')]
  vcf_UTR[, APA_info := paste(gene, strand, iso_loc, number_isos, iso_region, sep='_')]
  vcf_UTR[, PAS_info := ifelse(PAS_1<51, 'PAS_proximal', '')]
  vcf_UTR[, c('chrom', 'chromStart', 'chromEnd', 'isoStart', 'isoStop', 'gene',
              'strand', 'iso_loc', 'number_isos', 'iso_region', 'UTRstart', 'UTRstop',
              'pr_eqtl', 'pr_gwas', 'cons', 'PAS', 'PAS_1', 'stop_d', 'region',
              'in_eclip', 'in_miR', 'ref', 'alt') := NULL]
  #collapse (currently multiple entries for same variant that are in/near more than one element)
  unique_vars<-unique(vcf_UTR$var_id)
  compressed_variants<-data.table::data.table()
  for (n in 1:length(unique_vars)) {
    matches<-vcf_UTR[var_id==unique_vars[n],]
    miR_info<-paste(matches$miR_info, collapse = "|")
    gwas_info<-paste(matches$gwas_info, collapse = "|")
    clinvar_info<-paste(matches$clinvar_info, collapse = "|")
    matches[, c('miR_info', 'gwas_info', 'clinvar_info') := NULL]
    matches<-unique(matches)
    row<-cbind(matches, miR_info, gwas_info, clinvar_info)
    compressed_variants<-rbind(compressed_variants, row)
  }
  #rezip everything
  system('gzip *.bed')
  system('gzip *.fa')
  #write output
  setwd(path_to_filename)
  data.table::fwrite(compressed_variants, paste('processed_', filename, sep=''), sep = "\t", quote = FALSE, col.names = TRUE, row.names = FALSE)
  #clean workspace
  rm(list = ls())
}
