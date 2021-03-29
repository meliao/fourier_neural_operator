for i in plots preds models results
do
  if [[ ! -d $i ]]; then
    mkdir $i
  fi
done

python fourier_1d.py \
--data_fp ~/projects/fourier_neural_operator/data/2021-03-17_training_Burgers_data_GRF1.mat \
--model_fp models/first_run \
--preds_fp preds/first_run.mat \
--results_fp results/first_run.txt \
--epochs 3 \
--freq_modes 4
