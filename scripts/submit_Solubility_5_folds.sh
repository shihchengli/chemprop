#!/bin/bash

chemprop_dir=~/packages/chemprop  # location of chemprop directory, CHANGE ME

dataset=Solubility
train_path=../data/$dataset.csv


# Model training
# 1. ensembles
results_dir=../models/5_folds/random/$dataset/ensembles
python $chemprop_dir/train.py \
--dataset_type regression \
--data_path $train_path \
--save_preds \
--save_smiles_splits \
--save_dir $results_dir \
--epochs 100 \
--gpu 0 \
--cache_cutoff inf \
--ensemble_size 5 \
--num_folds 5

# 2. dropout
results_dir=../models/5_folds/random/$dataset/dropout
python $chemprop_dir/train.py \
--dataset_type regression \
--data_path $train_path \
--save_preds \
--save_smiles_splits \
--save_dir $results_dir \
--epochs 100 \
--gpu 0 \
--dropout 0.1 \
--cache_cutoff inf \
--num_folds 5

# 3. mve
results_dir=../models/5_folds/random/$dataset/mve
python $chemprop_dir/train.py \
--dataset_type regression \
--data_path $train_path \
--loss_function mve \
--save_preds \
--save_smiles_splits \
--epochs 100 \
--save_dir $results_dir \
--gpu 0 \
--cache_cutoff inf \
--num_folds 5

# 4. evidential
results_dir=../models/5_folds/random/$dataset/evidential_0.2
python $chemprop_dir/train.py \
--dataset_type regression \
--data_path $train_path \
--loss_function evidential \
--save_preds \
--save_smiles_splits \
--epochs 100 \
--save_dir $results_dir \
--gpu 0 \
--cache_cutoff inf \
--num_folds 5 \
--evidential_regularization 0.2

# 5. quantile regression
for alpha in 0.1 0.2 0.3 0.4 0.5
do
    results_dir=../models/5_folds/random/$dataset/conformal_quantile_regression/alpha_$alpha
    python $chemprop_dir/train.py \
    --dataset_type regression \
    --data_path $train_path \
    --save_preds \
    --save_smiles_splits \
    --epochs 100 \
    --save_dir $results_dir \
    --gpu 0 \
    --cache_cutoff inf \
    --loss_function quantile_interval \
    --quantile_loss_alpha $alpha \
    --num_folds 5
done

# Prediction
for i in 0 1 2 3 4
do
    # 1. ensemble
    results_dir=../models/5_folds/random/$dataset/ensembles
    python $chemprop_dir/predict.py \
    --test_path $results_dir/fold_$i/test_full.csv \
    --checkpoint_dir $results_dir/fold_$i \
    --preds_path $results_dir/fold_$i/ensemble_unc_preds.csv \
    --uncertainty_method ensemble \
    --evaluation_methods nll spearman miscalibration_area \
    --evaluation_scores_path $results_dir/fold_$i/ensemble_unc_evaluation_scores.csv

    # 2. dropout
    results_dir=../models/5_folds/random/$dataset/dropout
    python $chemprop_dir/predict.py \
    --test_path $results_dir/fold_$i/test_full.csv \
    --checkpoint_dir $results_dir/fold_$i/model_0/ \
    --preds_path $results_dir/fold_$i/dropout_unc_preds.csv \
    --uncertainty_method dropout \
    --uncertainty_dropout_p 0.1 \
    --dropout_sampling_size 10 \
    --evaluation_methods nll spearman miscalibration_area \
    --evaluation_scores_path $results_dir/fold_$i/dropout_unc_evaluation_scores.csv

    # 3. mve
    results_dir=../models/5_folds/random/$dataset/mve
    python $chemprop_dir/predict.py \
    --test_path $results_dir/fold_$i/test_full.csv \
    --checkpoint_dir $results_dir/fold_$i \
    --preds_path $results_dir/fold_$i/mve_unc_preds.csv \
    --uncertainty_method mve \
    --evaluation_methods nll spearman miscalibration_area \
    --evaluation_scores_path $results_dir/fold_$i/mve_unc_evaluation_scores.csv

    # 4. evidential
    results_dir=../models/5_folds/random/$dataset/evidential_0.2
    for uncertainty_method in evidential_total evidential_epistemic evidential_aleatoric
    do
        python $chemprop_dir/predict.py \
        --test_path $results_dir/fold_$i/test_full.csv \
        --checkpoint_dir $results_dir/fold_$i \
        --preds_path $results_dir/fold_$i/$uncertainty_method\_unc_preds.csv \
        --uncertainty_method $uncertainty_method \
        --evaluation_methods nll spearman miscalibration_area \
        --evaluation_scores_path $results_dir/fold_$i/$uncertainty_method\_unc_evaluation_scores.csv
    done

    # 5. quantile regression
    for alpha in 0.1 0.2 0.3 0.4 0.5
    do
        results_dir=../models/5_folds/random/$dataset/conformal_quantile_regression/alpha_$alpha/
        python $chemprop_dir/predict.py \
        --test_path $results_dir/fold_$i/test_full.csv \
        --checkpoint_dir $results_dir/fold_$i \
        --preds_path $results_dir/fold_$i/conformal_unc_preds.csv
    done

    # 6. conformal quantile regression
    for alpha in 0.1 0.2 0.3 0.4 0.5
    do
        results_dir=../models/5_folds/random/$dataset/conformal_quantile_regression/alpha_$alpha/
        python $chemprop_dir/predict.py \
        --test_path $results_dir/fold_$i/test_full.csv \
        --calibration_path $results_dir/fold_$i/val_full.csv \
        --calibration_method conformal_quantile_regression \
        --conformal_alpha $alpha \
        --evaluation_methods conformal_coverage \
        --evaluation_scores_path $results_dir/fold_$i/cqr_unc_eval.csv \
        --checkpoint_dir $results_dir/fold_$i \
        --preds_path $results_dir/fold_$i/cqr_unc_preds.csv
    done
done
