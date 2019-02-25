META_TRAIN := ../input/metadata_train.csv
TRAIN := ../input/train.parquet

FEAT_DIR := ../features/tsfresh-features/init_parameters.json
FEATNAME := train_fresh_0.pkl
FEATURES := ${FEAT_DIR}/${FEATNAME}

HIDDEN := 128
LINEAR := 100
NATTEN := 50
ANNEAL := --anneal
TBATCH := 512
VBATCH := 512
NSPLITS := 5
SEED := 42
ENBL_TEST := --enable_local_test
TEST_SIZE := 0.3
DEVICE := "cpu"
NEPOCHS := 50

train: ${FEATURES}
	python train.py \
	--features ${FEATURES} \
	--metadata ${META_TRAIN} \
	--hidden_size ${HIDDEN} \
	--linear_size ${LINEAR} \
	--n_attention ${NATTEN} \
	${ANNEAL} \
	--train_batch ${TBATCH} \
	--val_batch ${VBATCH} \
	--n_splits ${NSPLITS} \
	--seed ${SEED} \
	${ENBL_TEST} \
	--test_size ${TEST_SIZE} \
	--device ${DEVICE} \
	--n_epochs ${NEPOCHS}

${FEATURES}: ${TRAIN} ${META_TRAIN}
	make -C ../feature-calcuration ${FEATURES} DATA=${TRAIN} METADATA=${META_TRAIN} FEATNAME=${FEATNAME}
