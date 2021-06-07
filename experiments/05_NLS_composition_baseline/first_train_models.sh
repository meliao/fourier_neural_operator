for i in [ preds models results ]
do
  mkdir ~/projects/fourier_neural_operator/experiments/05_NLS_composition_baseline/${i}
done

python -m experiments.05_NLS_composition_baseline.train_models \
--data_fp data/2021-06-07_NLS_two_step_training_dataset.mat \
--model_fp experiments/05_NLS_composition_baseline/models/first_model \
--preds_fp experiments/05_NLS_composition_baseline/preds/first_model.mat \
--results_fp experiments/05_NLS_composition_baseline/results/first_model.txt \
--freq_modes 2 \
--epochs 5
