#!/bin/bash

chemprop_dir=~/packages/chemprop  # location of chemprop directory, CHANGE ME

dataset=stokes_primary_regr

for i in 0 1 2 3 4 5 6 7 8 9
do
    train_path=../data/$dataset/fold_$i/train.csv
    val_path=../data/$dataset/fold_$i/val.csv
    # Model training
    # 1. ensembles
    results_dir=../models/5_folds/random/$dataset/ensembles/$i
    python $chemprop_dir/train.py \
    --dataset_type regression \
    --data_path $train_path \
    --separate_val_path $val_path \
    --separate_test_path $val_path \
    --save_preds \
    --target_columns Mean_Inhibition \
    --features_generator rdkit_2d_normalized \
    --no_features_scaling \
    --save_dir $results_dir \
    --epochs 30 \
    --gpu 0 \
    --cache_cutoff inf \
    --extra_metrics mae \
    --ensemble_size 5

    # 2. dropout
    results_dir=../models/5_folds/random/$dataset/dropout/$i
    python $chemprop_dir/train.py \
    --dataset_type regression \
    --data_path $train_path \
    --separate_val_path $val_path \
    --separate_test_path $val_path \
    --save_preds \
    --target_columns Mean_Inhibition \
    --features_generator rdkit_2d_normalized \
    --no_features_scaling \
    --epochs 30 \
    --save_dir $results_dir \
    --gpu 0 \
    --dropout 0.1 \
    --extra_metrics mae \
    --cache_cutoff inf

    # 3. mve
    results_dir=../models/5_folds/random/$dataset/mve/$i
    python $chemprop_dir/train.py \
    --dataset_type regression \
    --data_path $train_path \
    --separate_val_path $val_path \
    --separate_test_path $val_path \
    --save_preds \
    --target_columns Mean_Inhibition \
    --features_generator rdkit_2d_normalized \
    --no_features_scaling \
    --loss_function mve \
    --epochs 30 \
    --save_dir $results_dir \
    --gpu 0 \
    --extra_metrics mae \
    --cache_cutoff inf

    # 4. evidential
    results_dir=../models/5_folds/random/$dataset/evidential_0.1/$i
    python $chemprop_dir/train.py \
    --dataset_type regression \
    --data_path $train_path \
    --separate_val_path $val_path \
    --separate_test_path $val_path \
    --save_preds \
    --target_columns Mean_Inhibition \
    --features_generator rdkit_2d_normalized \
    --no_features_scaling \
    --loss_function evidential \
    --epochs 30 \
    --save_dir $results_dir \
    --gpu 0 \
    --cache_cutoff inf \
    --extra_metrics mae \
    --evidential_regularization 0.1

    # 5. quantile regression
    for alpha in 0.3
    do
        results_dir=../models/5_folds/random/$dataset/conformal_quantile_regression/alpha_$alpha/$i
        python $chemprop_dir/train.py \
        --dataset_type regression \
        --data_path $train_path \
        --separate_val_path $val_path \
        --separate_test_path $val_path \
        --save_preds \
        --target_columns Mean_Inhibition \
        --features_generator rdkit_2d_normalized \
        --no_features_scaling \
        --epochs 30 \
        --save_dir $results_dir \
        --gpu 0 \
        --cache_cutoff inf \
        --loss_function quantile_interval \
        --quantile_loss_alpha $alpha
    done
done

# Prediction
test_path=../data/broad_smiles_validated_full.csv
for i in 0 1 2 3 4 5 6 7 8 9
do
    val_path=../data/$dataset/fold_$i/val.csv
    # 1. ensemble
    results_dir=../models/5_folds/random/$dataset/ensembles
    python $chemprop_dir/predict.py \
    --test_path $test_path \
    --smiles_columns  SMILES \
    --features_generator rdkit_2d_normalized \
    --no_features_scaling \
    --checkpoint_dir $results_dir/$i/fold_0 \
    --preds_path $results_dir/$i/fold_0/ensemble_unc_preds.csv \
    --no_cuda \
    --uncertainty_method ensemble \
    --evaluation_methods nll spearman miscalibration_area \
    --evaluation_scores_path $results_dir/$i/fold_0/ensemble_unc_evaluation_scores.csv

    # 2. dropout
    results_dir=../models/5_folds/random/$dataset/dropout
    python $chemprop_dir/predict.py \
    --test_path $test_path \
    --smiles_columns  SMILES \
    --features_generator rdkit_2d_normalized \
    --no_features_scaling \
    --checkpoint_dir $results_dir/$i/fold_0/model_0/ \
    --preds_path $results_dir/$i/fold_0/dropout_unc_preds.csv \
    --uncertainty_method dropout \
    --uncertainty_dropout_p 0.1 \
    --dropout_sampling_size 10 \
    --no_cuda \
    --evaluation_methods nll spearman miscalibration_area \
    --evaluation_scores_path $results_dir/$i/fold_0/dropout_unc_evaluation_scores.csv

    # 3. mve
    results_dir=../models/5_folds/random/$dataset/mve
    python $chemprop_dir/predict.py \
    --test_path $test_path \
    --smiles_columns  SMILES \
    --features_generator rdkit_2d_normalized \
    --no_features_scaling \
    --checkpoint_dir $results_dir/$i/fold_0 \
    --preds_path $results_dir/$i/fold_0/mve_unc_preds.csv \
    --no_cuda \
    --uncertainty_method mve \
    --evaluation_methods nll spearman miscalibration_area \
    --evaluation_scores_path $results_dir/$i/fold_0/mve_unc_evaluation_scores.csv

    # 4. evidential
    results_dir=../models/5_folds/random/$dataset/evidential_0.1
    for uncertainty_method in evidential_total evidential_epistemic evidential_aleatoric
    do
        python $chemprop_dir/predict.py \
        --test_path $test_path \
        --smiles_columns  SMILES \
        --features_generator rdkit_2d_normalized \
        --no_features_scaling \
        --checkpoint_dir $results_dir/$i/fold_0 \
        --preds_path $results_dir/$i/fold_0/$uncertainty_method\_unc_preds.csv \
        --uncertainty_method $uncertainty_method \
        --no_cuda \
        --evaluation_methods nll spearman miscalibration_area \
        --evaluation_scores_path $results_dir/$i/fold_0/$uncertainty_method\_unc_evaluation_scores.csv
    done

    # 5. quantile regression
    for alpha in 0.3
    do
        results_dir=../models/5_folds/random/$dataset/conformal_quantile_regression/alpha_$alpha/
        python $chemprop_dir/predict.py \
        --test_path $test_path \
        --smiles_columns  SMILES \
        --features_generator rdkit_2d_normalized \
        --no_features_scaling \
        --no_cuda \
        --checkpoint_dir $results_dir/$i/fold_0 \
        --preds_path $results_dir/$i/fold_0/conformal_unc_preds.csv
    done

    # 6. conformal quantile regression
    for alpha in 0.3
    do
        results_dir=../models/5_folds/random/$dataset/conformal_quantile_regression/alpha_$alpha/
        python $chemprop_dir/predict.py \
        --test_path $test_path \
        --smiles_columns  SMILES \
        --features_generator rdkit_2d_normalized \
        --no_features_scaling \
        --no_cuda \
        --calibration_path $val_path \
        --calibration_method conformal_quantile_regression \
        --conformal_alpha $alpha \
        --evaluation_methods conformal_coverage \
        --evaluation_scores_path $results_dir/$i/fold_0/cqr_unc_eval.csv \
        --checkpoint_dir $results_dir/$i/fold_0 \
        --preds_path $results_dir/$i/fold_0/cqr_unc_preds.csv
    done
done
