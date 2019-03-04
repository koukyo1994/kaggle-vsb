for tag in `ls trainer`; do \
  python test.py --features ../features/basic-features/160d/basic_test.pkl --tag $tag; \
done

python ensemble.py

kaggle competitions submit vsb-power-line-fault-detection -f submission/ensemble.csv -m "ensemble of over 50 pred from lstm-attention with basic features with scaling"
