META_TRAIN := ../input/metadata_train.csv
TRAIN := ../input/train.parquet
FEAT_DIR := ../features
FEATURES := ${FEAT_DIR}/train_fresh_0.pkl

train: ${FEATURES}
	python train.py --input ${FEATURES}

${FEATURES}: ${TRAIN} ${META_TRAIN}
	python core/feature_extraction.py --path ${TRAIN} --name ${FEAT_DIR}
