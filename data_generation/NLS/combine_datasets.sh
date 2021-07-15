python ~/projects/fourier_neural_operator/data_generation/combine_datasets.py \
-in_dir /share/data/willett-group/meliao/data/2021-07-14_NLS_data_05/ \
-out_fp /share/data/willett-group/meliao/data/2021-07-14_NLS_data_05.mat \
-key t x output \
-train_split 1000 /share/data/willett-group/meliao/data/2021-07-14_NLS_data_05_train.mat \
-test_split 100 /share/data/willett-group/meliao/data/2021-07-14_NLS_data_05_test.mat