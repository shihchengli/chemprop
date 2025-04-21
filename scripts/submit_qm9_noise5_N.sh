#!/bin/bash

chemprop_dir=~/packages/chemprop  # location of chemprop directory, CHANGE ME

dataset=noise5_N
train_path=../data/qm9/noise5_N/qm9_noise5_N.csv

# Model training
# 1. ensembles
for i in 0 1 2 3 4
do
    results_dir=../models/$dataset/ensembles/$i
    python $chemprop_dir/train.py \
    --dataset_type regression \
    --data_path $train_path \
    --save_preds \
    --save_dir $results_dir \
    --hidden_size 1000 \
    --depth 4 \
    --ffn_num_layers 2 \
    --aggregation norm \
    --epochs 50 \
    --gpu 0 \
    --cache_cutoff 10000 \
    --pytorch_seed $i
done

# 2. dropout
results_dir=../models/$dataset/dropout
python $chemprop_dir/train.py \
--dataset_type regression \
--data_path $train_path \
--save_preds \
--hidden_size 1000 \
--depth 4 \
--ffn_num_layers 2 \
--aggregation norm \
--epochs 50 \
--save_dir $results_dir \
--gpu 0 \
--dropout 0.1 \
--cache_cutoff 10000

# 3. mve
results_dir=../models/$dataset/mve
python $chemprop_dir/train.py \
--dataset_type regression \
--data_path $train_path \
--loss_function mve \
--save_preds \
--hidden_size 1000 \
--depth 4 \
--ffn_num_layers 2 \
--aggregation norm \
--epochs 50 \
--save_dir $results_dir \
--gpu 0 \
--cache_cutoff 10000

# 4. evidential
results_dir=../models/$dataset/evidential_0.2
python $chemprop_dir/train.py \
--dataset_type regression \
--data_path $train_path \
--loss_function evidential \
--save_preds \
--hidden_size 1000 \
--depth 4 \
--ffn_num_layers 2 \
--aggregation norm \
--epochs 50 \
--save_dir $results_dir \
--gpu 0 \
--cache_cutoff 10000 \
--evidential_regularization 0.2

# 5. quantile regression
for alpha in 0.3
do
    results_dir=../models/$dataset/conformal_quantile_regression/alpha_$alpha
    python $chemprop_dir/train.py \
    --dataset_type regression \
    --data_path $train_path \
    --save_preds \
    --save_smiles_splits \
    --hidden_size 1000 \
    --depth 4 \
    --ffn_num_layers 2 \
    --aggregation norm \
    --epochs 50 \
    --save_dir $results_dir \
    --gpu 0 \
    --cache_cutoff 10000 \
    --loss_function quantile_interval \
    --quantile_loss_alpha $alpha
done

i=0
# Prediction
# 1. ensemble
results_dir=../models/$dataset/ensembles
python $chemprop_dir/predict.py \
--test_path $results_dir/fold_$i/test_full.csv \
--checkpoint_dir $results_dir \
--preds_path $results_dir/ensemble_unc_preds.csv \
--uncertainty_method ensemble \
--evaluation_methods nll spearman miscalibration_area \
--evaluation_scores_path $results_dir/ensemble_unc_evaluation_scores.csv

# 2. dropout
results_dir=../models/$dataset/dropout
python $chemprop_dir/predict.py \
--test_path $results_dir/fold_$i/test_full.csv \
--checkpoint_dir $results_dir \
--preds_path $results_dir/dropout_unc_preds.csv \
--uncertainty_method dropout \
--uncertainty_dropout_p 0.1 \
--dropout_sampling_size 10 \
--evaluation_methods nll spearman miscalibration_area \
--evaluation_scores_path $results_dir/dropout_unc_evaluation_scores.csv

# 3. mve
results_dir=../models/$dataset/mve
python $chemprop_dir/predict.py \
--test_path $results_dir/fold_$i/test_full.csv \
--checkpoint_dir $results_dir \
--preds_path $results_dir/mve_unc_preds.csv \
--uncertainty_method mve \
--evaluation_methods nll spearman miscalibration_area \
--evaluation_scores_path $results_dir/mve_unc_evaluation_scores.csv

# 4. evidential
results_dir=../models/$dataset/evidential_0.2
for uncertainty_method in evidential_total evidential_epistemic evidential_aleatoric
do
    python $chemprop_dir/predict.py \
    --test_path $results_dir/fold_$i/test_full.csv \
    --checkpoint_dir $results_dir \
    --preds_path $results_dir/$uncertainty_method\_unc_preds.csv \
    --uncertainty_method $uncertainty_method \
    --evaluation_methods nll spearman miscalibration_area \
    --evaluation_scores_path $results_dir/$uncertainty_method\_unc_evaluation_scores.csv
done

# 5. quantile regression
for alpha in 0.3
do
    results_dir=../models/$dataset/conformal_quantile_regression/alpha_$alpha/
    python $chemprop_dir/predict.py \
    --test_path $results_dir/fold_$i/test_full.csv \
    --checkpoint_dir $results_dir \
    --preds_path $results_dir/conformal_unc_preds.csv
done

# 6. conformal quantile regression
for alpha in 0.3
do
    results_dir=../models/$dataset/conformal_quantile_regression/alpha_$alpha/
    python $chemprop_dir/predict.py \
    --test_path $results_dir/fold_$i/test_full.csv \
    --calibration_path $results_dir/fold_0/val_full.csv \
    --calibration_method conformal_quantile_regression \
    --conformal_alpha $alpha \
    --evaluation_methods conformal_coverage \
    --evaluation_scores_path $results_dir/cqr_unc_eval.csv \
    --checkpoint_dir $results_dir \
    --preds_path $results_dir/cqr_unc_preds.csv
done
