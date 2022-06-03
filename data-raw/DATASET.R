## code to prepare `DATASET`
miR_predictions<-data.table::fread('hg38_miR_predictions_final.txt.gz') #conserved only
miR_predictions<-miR_predictions[, c('mergekey', 'miR_family', 'seed_match', 'Pct', 'strand', 'context_pile', 'cons', 'i.cons')]
#miR_predictions[, fam_con := data.table::tstrsplit(i.cons, '_')[[1]]]
#miR_predictions<-miR_predictions[fam_con=='broadConsFam' | fam_con=='consFam']

cadd_gd<-data.table::fread('all_gd_UTR_vars.bed.gz')

#GLM for gwas/eqtl
all_vcf<-data.table::fread('GTEx_v8_finemapping_DAPG_scottUTR_processed.txt.gz')
#APA info
poly_info <- data.table::fread('../CharVar/inst/extdata/all_APA_peak_coords_hg38.bed.gz')
names(poly_info)<-c('chrom', 'isoStart', 'isoStop', 'gene', 'score', 'strand', 'number_isos', 'iso_loc', 'UTRstart', 'UTRstop')
poly_info[, UTR_length:=UTRstop-UTRstart]
UTR_lengths<-unique(poly_info[, c('gene', 'number_isos', 'UTR_length', 'UTRstart')])
data.table::setkey(UTR_lengths, gene)
data.table::setkey(all_vcf, gene)
all_vcf<-UTR_lengths[all_vcf]
all_vcf[, var_info_tot:=paste(var_info, ref, alt, sep='_')]
int<-data.table::fread('scott_vcf.bed.3pseq_intersected.txt.gz')
int[, V5:=NULL]
names(int)<-c('chrom', 'chromStart', 'chromStop', 'var_info_tot', 'isoStart', 'isoStop', 'gene', 'score', 'strand', 'number_isos', 'iso_loc', 'UTRstart', 'UTRstop')
int<-unique(int[, c('var_info_tot', 'iso_loc', 'isoStop', 'isoStart')])
data.table::setkey(int, var_info_tot)
data.table::setkey(all_vcf, var_info_tot)
all_vcf<-int[all_vcf]
rm(int, poly_info)
all_vcf$iso_loc<-as.numeric(all_vcf$iso_loc)
all_vcf$dvp<-all_vcf$iso_loc/all_vcf$number_isos
#add distance from closest PAsite
all_vcf[dvp<=1,inter_PAS:=isoStop-isoStart]
all_vcf[,PAS_1:=abs(chromEnd-isoStart)]
all_vcf[,PAS_2:=abs(chromEnd-isoStop)]
all_vcf[PAS_1<PAS_2, PAS_1:=PAS_1]
all_vcf[PAS_1>=PAS_2, PAS_1:=PAS_2]
all_vcf[,PAS_2:=NULL]
all_vcf[, stop_d:=abs(chromStart-UTRstart)]
all_vcf[, var_id := paste(base_id, ref, alt, sep='_')]
#add miR conservation info
all_vcf_miRs<-data.table::fread("GTEx_v8_finemapping_DAPG_scottUTR_processed_expanded_miR_info.txt.gz", header=TRUE)
all_vcf_miRs[, var_id := paste(chrom, chromStart, chromEnd, ref, alt, sep='_')]
all_vcf_miRs[, c('chrom1', 'miR_chromStart', 'miR_chromEnd', 'seed_match', 'Pct',
                 'context_pile', 'fam_con', 'site_con'):=data.table::tstrsplit(site_info, '_')]
cons_miRs<-all_vcf_miRs[fam_con=='consFam' | fam_con=='broadConsFam']$var_id
all_vcf[, var_id := paste(base_id, ref, alt, sep='_')]

#GWAS
gwas_all<-data.table::fread('gwas_scottUTRvars_processed.txt.gz') #includes all phenotypes
gwas_all[, iso_loc:= data.table::tstrsplit(pseq_UTR_info, '_')[5]]
gwas_all$iso_loc<-as.numeric(gwas_all$iso_loc)
gwas_all[, iso_number:= data.table::tstrsplit(pseq_UTR_info, '_')[4]]
gwas_all$iso_number<-as.numeric(gwas_all$iso_number)
gwas_all[, UTRstart:= data.table::tstrsplit(pseq_UTR_info, '_')[7]]
gwas_all$UTRstart<-as.numeric(gwas_all$UTRstart)
gwas_all[, UTRstop:= data.table::tstrsplit(pseq_UTR_info, '_')[8]]
gwas_all$UTRstop<-as.numeric(gwas_all$UTRstop)
gwas_all[, dvp:= iso_loc/iso_number]
gwas_all[dvp>1, dvp := NA] #remove variants past last isoform
gwas_all[, UTR_length := UTRstop-UTRstart]
gwas_all[, stop_d:=abs(chromStart-UTRstart)] #distance of SNV from canonical stop codon
gwas_pseq<-unique(gwas_all[, c('base_id', 'dvp', 'iso_loc', 'iso_number', 'pip', 'UTR_length')])
#add distance from closest PAsite
gwas_all[dvp<=1,inter_PAS:=isoStop-isoStart]
gwas_all[,PAS_1:=abs(chromEnd-isoStart)]
gwas_all[,PAS_2:=abs(chromEnd-isoStop)]
gwas_all[PAS_1<PAS_2, PAS_1:=PAS_1]
gwas_all[PAS_1>=PAS_2, PAS_1:=PAS_2]
gwas_all[,PAS_2:=NULL]
gwas_all[, stop_d:=abs(chromStart-UTRstart)]
gwas_all[, var_id := paste(base_id, ref, alt, sep='_')]
gwas_all[, c('chrom1', 'miR_chromStart', 'miR_chromEnd', 'seed_match', 'Pct',
             'context_pile', 'fam_con', 'site_con'):=data.table::tstrsplit(site_info, '_')]

##eQTL:
#define categorical variables that may affect pip
all_vcf[,in_miR:=ifelse(var_id %in% cons_miRs, 1, 0)]
all_vcf[, PAS := ifelse(PAS_1<=50, 1, 0)]
all_vcf[, region := ifelse(number_isos==1 | dvp<=0.5, 1, 0)]
all_vcf$phastcons_100<-as.numeric(all_vcf$phastcons_100)
all_vcf[, cons := ifelse(phastcons_100>0.5, 1, 0)]
all_vcf$pip<-as.numeric(all_vcf$pip)
all_vcf[, pip_model := ifelse(pip>=0.5, 1, 0)]
all_vcf[, in_eclip := ifelse(eclip_tot!='', 1, 0)]
#model
#split data for training
train_vcf<-rbind(all_vcf[pip>0.95], all_vcf[pip<0.001])
train_vcf<-na.omit(train_vcf, cols=c('in_eclip', 'PAS_1', 'in_miR', 'phastcons_100', 'region', 'number_isos'))
test_vcf<-all_vcf[pip>0.001 & pip<0.95,]

eqtl_model<-glm(pip_model~in_eclip+PAS+in_miR+cons+region+number_isos,
                family  = binomial, data=train_vcf)

##GWAS:
#define categorical variables that may affect pip
names(gwas_all)[names(gwas_all)=='iso_number']<-'number_isos'
gwas_all[,in_miR:=ifelse(fam_con=='consFam' | fam_con=='broadConsFam', 1, 0)]
gwas_all[is.na(in_miR), in_miR := 0]
gwas_all[, PAS := ifelse(PAS_1<=50, 1, 0)]
gwas_all[, region := ifelse(number_isos==1 | dvp<=0.5, 1, 0)]
gwas_all$phastcons_100<-as.numeric(gwas_all$phastcons_100)
gwas_all[, cons := ifelse(phastcons_100>0.5, 1, 0)]
gwas_all$pip<-as.numeric(gwas_all$pip)
gwas_all[, pip_model := ifelse(pip>=0.5, 1, 0)]
gwas_all[, in_motif := ifelse(RBPs!='', 1, 0)]
gwas_all[, in_eclip := ifelse(eclip_tot!='', 1, 0)]
#model
#split data for training
gwas_all[, sample_key := seq(1:nrow(gwas_all))]
train_gwas<-gwas_all[sample_key %in% sample(gwas_all$sample_key, 270000),]
test_gwas<-gwas_all[!sample_key %in% train_gwas$sample_key,]
train_gwas<-na.omit(train_gwas, cols=c('in_eclip', 'PAS_1', 'in_miR', 'phastcons_100', 'region', 'number_isos'))

#model
gwas_model<-glm(pip_model~in_eclip+PAS+in_miR+cons+region+number_isos,
                family  = binomial, data=train_gwas)

##write system data (in CharVar dir)
usethis::use_data(miR_predictions, cadd_gd, eqtl_model, gwas_model, overwrite=TRUE, compress = 'gzip', internal = TRUE)

