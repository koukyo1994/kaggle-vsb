META_TRAIN := ../input/metadata_train.csv
TRAIN := ../input/train.parquet
FEAT_DIR := ../features
FEATURES := ${FEAT_DIR}/train_fresh_0.pkl

train: ${FEATURES}
	python train.py --input ${FEATURES}

${FEATURES}: ${TRAIN} ${META_TRAIN}
	make -C ../feature-calcuration DATA=${TRAIN} METADATA=${META_TRAIN} FEATURES=${FEATURES}
