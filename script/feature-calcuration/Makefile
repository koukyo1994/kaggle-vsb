METADATA := ../input/metadata_train.csv
DATA := ../input/train.parquet
FEATURES := train_fresh_0.pkl
NCHUNK := 24
NJOBS := 2
PARAMS := tsfresh-features/init_parameters.json

tsfresh: ${METADATA} ${DATA}
	python tsfresh-features/main.py --metadata ${METADATA} --path ${DATA} --name ${FEATURES} --nchunk ${NCHUNK} --n_jobs ${NJOBS} --parameters ${PARAMS}
