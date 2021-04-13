for i in models preds results plots jobs logs
do
  if [[ ! -d experiments/03_use_Haar_DWT/$i ]]; then
    mkdir experiments/03_use_Haar_DWT/$i
  fi
done

python -m experiments.03_use_Haar_DWT.run_haarDWT_network \
 --data_fp data/2021-03-17_training_Burgers_data_GRF1.mat \
 --model_fp experiments/03_use_Haar_DWT/models/first_run \
 --preds_fp experiments/03_use_Haar_DWT/preds/first_run.mat \
 --results_fp experiments/03_use_Haar_DWT/results/first_run.txt \
 --epochs 20 \
 --keep 1024 \
 --width 4
