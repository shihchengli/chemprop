#!/bin/bash
# Lower-N datasets
nohup bash submit_freesolv_5_folds.sh > freesolv_random.log 2>&1 &
nohup bash submit_delaney_5_folds.sh > delaney_random.log 2>&1 &
nohup bash submit_lipo_5_folds.sh > lipo_random.log 2>&1 &
nohup bash submit_qm7_5_folds.sh > qm7_random.log 2>&1 &

# Larger-N datasets
nohup bash submit_qm9_5_folds.sh > qm9_random.log 2>&1 &
nohup bash submit_qm9_5_folds_co_tranining.sh > qm9_random_co_tranining.log 2>&1 &
nohup bash submit_enamine_5_folds.sh > enamine_random.log 2>&1 &

# TDC datasets
nohup bash submit_clearance_hepatocyte_az_5_folds.sh > clearance_hepatocyte_az.log 2>&1 &
nohup bash submit_ld50_zhu_5_folds.sh > ld50_zhu_random.log 2>&1 &
nohup bash submit_ppbr_az_5_folds.sh > ppbr_az_random.log 2>&1 &

# ADME datasets from Biogen
nohup bash submit_HLM_5_folds.sh > HLM_random.log 2>&1 &
nohup bash submit_MDR1_MDCK_ER_5_folds.sh > MDR1_MDCK_ER_random.log 2>&1 &
nohup bash submit_Solubility_5_folds.sh > Solubility_random.log 2>&1 &
nohup bash submit_RLM_5_folds.sh > RLM_random.log 2>&1 &
nohup bash submit_hPPB_5_folds.sh > hPPB_random.log 2>&1 &
nohup bash submit_rPPB_5_folds.sh > rPPB_random.log 2>&1 &

# Noisy datasets
nohup bash submit_groupadditivity_noise20_half.sh > submit_groupadditivity_noise20_half.log 2>&1 &
nohup bash submit_groupadditivity_noise20_nitrogen.sh > submit_groupadditivity_noise20_nitrogen.log 2>&1 &
nohup bash submit_qm9_heavy_atoms.sh > submit_qm9_heavy_atoms.log 2>&1 &
nohup bash submit_qm9_values_split.sh > submit_qm9_values_split.log 2>&1 &

nohup bash submit_qm9_noise5.sh > submit_qm9_noise5.log 2>&1 &
nohup bash submit_qm9_noise5_N.sh > submit_qm9_noise5_N.log 2>&1 &
nohup bash submit_qm9_ignore_N.sh > submit_qm9_ignore_N.log 2>&1 &

# Stock
nohup bash submit_stokes_primary_regr_5_folds.sh > submit_stokes_primary_regr_5_folds.log 2>&1 &
