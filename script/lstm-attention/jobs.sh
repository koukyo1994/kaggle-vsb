for i in `seq 0 50`
do
  python train.py --features ../features/basic-features/160d/basic_train.pkl --metadata ../input/metadata_train.csv --hidden_size 128 --linear_size 64 --n_attention 50 --train_batch 128 --val_batch 512 --n_splits 5 --seed $i --test_size 0.3 --device "cpu" --n_epochs 50 --enable_local_test --scaling
done
