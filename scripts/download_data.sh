#!/bin/sh

mkdir data
cd data
mkdir relative
cd relative

# Image pairs from IMC 2021
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_lincoln_memorial_statue.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_piazza_san_marco.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_london_bridge.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_sagrada_familia.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_british_museum.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_milan_cathedral.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_st_pauls_cathedral.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_florence_cathedral_side.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/imc_mount_rushmore.h5

# Image pairs from ScanNet. Evaluation protocol from SuperGlue paper (Sarlin et al.)
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_spsg.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_sift.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_roma.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_dkm.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_aspanformer.h5

# Image pairs from MegaDepth
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_sift.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_spsg.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_splg.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_roma.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_dkm.h5
wget --no-check-certificate -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_aspanformer.h5