from sklearn.model_selection import StratifiedKFold


class Trainer:
    def __init__(self, train, test, n_splits=5, seed=42):
        self.train_set = train
        self.test_sett = test

        self.fold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)


class NNTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(NNTrainer, self).__init__()
