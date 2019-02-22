META_TRAIN := ../input/metadata_train.csv
TRAIN := ../input/train.parquet

FEAT_DIR := ../features/tsfresh-features/init_parameters.json
FEATNAME := train_fresh_0.pkl
FEATURES := ${FEAT_DIR}/${FEATNAME}

HIDDEN := 128
LINEAR := 100
NATTEN := 50
ANNEAL := False
TBATCH := 512
VBATCH := 512
NSPLITS := 5
SEED := 42
ENBL_TEST := False
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
	--anneal ${ANNEAL} \
	--train_batch ${TBATCH} \
	--val_batch ${VBATCH} \
	--n_splits ${NSPLITS} \
	--seed ${SEED} \
	--enable_local_test ${ENBL_TEST} \
	--test_size ${TEST_SIZE} \
	--device ${DEVICE} \
	--n_epochs ${NEPOCHS}

${FEATURES}: ${TRAIN} ${META_TRAIN}
	make -C ../feature-calcuration DATA=${TRAIN} METADATA=${META_TRAIN} FEATURES=${FEATNAME}
