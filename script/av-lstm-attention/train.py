import sys
import pickle

from pathlib import Path

from argparse import ArgumentParser

if __name__ == "__main__":
    sys.path.append("../..")
    sys.path.append("..")
    sys.path.append("./")
    from model import LSTMAttentionNet
    from script.common.utils import get_logger
    from script.common.adversarial_trainer import NNTrainer

    parser = ArgumentParser()
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--linear_size", default=100, type=int)
    parser.add_argument("--n_attention", default=50, type=int)

    parser.add_argument("--train_batch", default=128, type=int)

    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--loc_lambda", default=0.1, type=float)

    parser.add_argument("--n_epochs", default=50, type=int)

    parser.add_argument("--train_set")
    parser.add_argument("--validation_set")

    args = parser.parse_args()

    logger = get_logger("av-lstm-attention", "av-lstm-attention")
    logger.info(
        f"hidden_size: {args.hidden_size}, linear_size: {args.linear_size}")
    logger.info(f"n_attention: {args.n_attention}")
    logger.info(f"train_batch: {args.train_batch}")
    logger.info(f"n_splits: {args.n_splits}, seed: {args.seed}")
    logger.info(f"loc_lambda: {args.loc_lambda}")
    logger.info(f"n_epochs: {args.n_epochs}")
    logger.info(f"train_set: {args.train_set}")
    logger.info(f"validation_set: {args.validation_set}")

    with open(args.train_set, "rb") as f:
        train_set = pickle.load(f)

    with open(args.validation_set, "rb") as f:
        validation_set = pickle.load(f)

    trainer = NNTrainer(
        LSTMAttentionNet,
        logger,
        validation_set,
        n_splits=5,
        seed=args.seed,
        loc_lambda=args.loc_lambda,
        device="cpu",
        train_batch=128,
        kwargs={
            "hidden_size": args.hidden_size,
            "linear_size": args.linear_size,
            "input_shape": train_set[0].shape,
            "n_attention": args.n_attention
        })

    trainer.fit(train_set[0], train_set[1], args.n_epochs)
    trainer_path = Path(f"trainer/{trainer.tag}")
    trainer_path.mkdir(parents=True, exist_ok=True)

    trainer.fold = None
    trainer.valid_set = None
    trainer.local_loader = None
    trainer.model = None
    trainer.loss_fn = None
    trainer.logger = None
    with open(trainer_path / "trainer.pkl", "wb") as f:
        pickle.dump(trainer, f)
