METADATA := ../input/metadata_train.csv
METATEST := ../input/metadata_test.csv

DATA := ../input/train.parquet
TEST := ../input/test.parquet

FEATDIR := ../features

# tsfresh parameters
TSFRESH := tsfresh-features
PARAMETER := init_parameters.json
PARAMS := tsfresh-features/init_parameters.json


# basic parameters
BASIC := basic-features
DIMS := 160d

# featname
FEATNAME := train_fresh_0.pkl
FEATNAME_TEST := test_fresh_0.pkl

NCHUNK := 24
NJOBS := 2
NDIMS := 160

${FEATDIR}/${TSFRESH}/${PARAMETER}/${TSFRESH_FEATNAME}: ${METADATA} ${DATA}
	python tsfresh-features/main.py --metadata ${METADATA} --path ${DATA} --name ${TSFRESH_FEATNAME} --nchunk ${NCHUNK} --n_jobs ${NJOBS} --parameters ${PARAMS}

${FEATDIR}/${TSFRESH}/${PARAMETER}/${TSFRESH_FEATNAME_TEST}: ${METATEST} ${TEST}
	python tsfresh-features/main.py --metadata ${METATEST} --path ${TEST} --name ${TSFRESH_FEATNAME_TEST} --nchunk ${NCHUNK} --n_jobs ${NJOBS} --parameters ${PARAMS}

${FEATDIR}/${BASIC}/${DIMS}/${BASIC_FEATNAME}: ${METADATA} ${DATA}
	python basic-features/main.py --metadata ${METADATA} --path ${DATA} --name ${BASIC_FEATNAME} --n_dims ${NDIMS} --nchunk ${NCHUNK}

${FEATDIR}/${BASIC}/${DIMS}/${BASIC_FEATNAME_TEST}: ${METATEST} ${TEST}
	python basic-features/main.py --metadata ${METATEST} --path ${TEST} --name ${BASIC_FEATNAME_TEST} --n_dims ${NDIMS} --nchunk ${NCHUNK}


