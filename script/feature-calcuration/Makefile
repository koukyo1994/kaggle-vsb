METADATA := ../input/metadata_train.csv
METATEST := ../input/metadata_test.csv
DATA := ../input/train.parquet
TEST := ../input/test.parquet
FEATURES := train_fresh_0.pkl
FEATURES_TEST := test_fresh_0.pkl
NCHUNK := 24
NJOBS := 2
NDIMS := 160
PARAMS := tsfresh-features/init_parameters.json

tsfresh-train: ${METADATA} ${DATA}
	python tsfresh-features/main.py --metadata ${METADATA} --path ${DATA} --name ${FEATURES} --nchunk ${NCHUNK} --n_jobs ${NJOBS} --parameters ${PARAMS}

tsfresh-test: ${METATEST} ${TEST}
	python tsfresh-features/main.py --metadata ${METATEST} --path ${TEST} --name ${FEATURES_TEST} --nchunk ${NCHUNK} --n_jobs ${NJOBS} --parameters ${PARAMS}

basic-train: ${METADATA} ${DATA}
	python basic-features/main.py --metadata ${METADATA} --path ${DATA} --name ${FEATURES} --n_dims ${NDIMS} --nchunk ${NCHUNK}
