# folding Nicole's pool

```
	/home/mjens/bin/RBPamp --format=fasta nsRBNS_oligos_taliaferro_et_al.fa --fold --adap5= --adap3= \
	--min-k=1 --max-k=12 --run-path=fold --debug=fold --overwrite -T22.5
```

# folding Bridget's pool
```
	/home/mjens/bin/RBPamp --format=fasta 3utrOligoPool_final_T7.fa --fold --adap5= --adap3= \
        --min-k=1 --max-k=12 --run-path=fold_bridget --debug=fold --overwrite -T4

```

join -j 1 <(sort input.counts) <(sort msi_25.counts) | join -j 1 /dev/stdin  <(sort msi_125.counts) | join -j 1 /dev/stdin <(sort msi_625.counts) | sed 's/ /\t/g' > msi1_matrix.csv
join -j 1 <(sort input.counts) <(sort rbfox2_25.counts) | join -j 1 /dev/stdin <(sort rbfox2_125.counts) | join -j 1 /dev/stdin <(sort rbfox2_625.counts) | sed 's/ /\t/g' > rbfox2_matrix.csv

# gather useful parameters for RBFOX1
cp /scratch/data/RBNS/RBFOX3/RBPamp/test/meanfield/7mer_affinities.tsv  rbfox3_psam_spa_grad_7mers.tsv
cat /scratch/data/RBNS/RBFOX3/RBPamp/seed_full/metrics/RBFOX3.R_value.7mer.tsv | sort | cut -f 1,2 > rbfox3_R_value_7mers.tsv

# MSI1
echo "from http://cisbp-rna.ccbr.utoronto.ca/bulk_archive.php Escores.txt" > msi1.rnacompete
cut -f 1,41 ~/RNAcompete/Escores.txt >> msi1.rnacompete 

cat /scratch/data/RBNS/MSI1/RBPamp/seeded/metrics/MSI1.R_value.7mer.tsv | sort | cut -f 1,2 > msi1_R_value_7mers.tsv
cp /scratch/data/RBNS/MSI1/RBPamp/seeded/affinity/7mer_affinities.tsv msi1_kmer_spa_7mer.tsv

# MBNL1
echo "from http://cisbp-rna.ccbr.utoronto.ca/bulk_archive.php Escores.txt" > mbnl1.rnacompete
cut -f 1,38 ~/RNAcompete/Escores.txt >> mbnl1.rnacompete 

join -j 1 <(sort input.counts) <(sort mbnl_25nM_140307BurB_D14-1668_NA_sequence.counts) | \
join -j 1 /dev/stdin  <(sort mbnl_125nM_140307BurB_D14-1668_NA_sequence.counts) | \
join -j 1 /dev/stdin <(sort mbnl_625nM_140307BurB_D14-1668_NA_sequence.counts) | sed 's/ /\t/g' > mbnl1_matrix.csv
