import os
import random
import time
import argparse
import json
from logger import Logger
from naive_r2i_functions import load_dataset_numpy, data_transform
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime

def set_random_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_rocket(model: str, n_kernels: int, seed: int):
    try:
        from sktime.transformations.panel.rocket import (
            MiniRocket,
            Rocket,
            MultiRocket,
            MiniRocketMultivariate,
            MultiRocketMultivariate,
        )
    except Exception as e:
        raise ImportError(
            "Missing dependency 'sktime'. Install it in this environment to run VROCKS."
        ) from e

    model = model.lower()
    if model == "rocket":
        return Rocket(num_kernels=n_kernels, random_state=seed, n_jobs=-1)
    if model == "minirocket":
        return MiniRocket(num_kernels=n_kernels, random_state=seed, n_jobs=-1)
    if model == "multirocket":
        return MultiRocket(num_kernels=n_kernels, random_state=seed, n_jobs=-1)
    if model == "multivariate_minirocket":
        return MiniRocketMultivariate(num_kernels=n_kernels, random_state=seed, n_jobs=-1)
    if model == "multivariate_multirocket":
        return MultiRocketMultivariate(num_kernels=n_kernels, random_state=seed, n_jobs=-1)

    raise ValueError(f"Unsupported model: {model}")


def is_multivariate_model(model: str) -> bool:
    return model.lower() in (
        "multivariate_minirocket",
        "multivariate_multirocket",
    )


def resolve_channel_handling(model: str, channel_handling: str | None) -> str:
    mv = is_multivariate_model(model)

    if channel_handling is None:
        return "separate" if mv else "concatenate"

    if mv and channel_handling != "separate":
        raise ValueError(
            f"Model '{model}' is multivariate. Use channel_handling='separate'."
        )

    if (not mv) and channel_handling == "separate":
        raise ValueError(
            f"Model '{model}' is univariate. Use 'concatenate' or "
            f"'concatenate_channelwise'."
        )

    return channel_handling

def main():
    parser = argparse.ArgumentParser(description="VROCKS baseline (ROCKET features + Ridge classifier)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "rocket",
            "minirocket",
            "multirocket",
            "multivariate_minirocket",
            "multivariate_multirocket",
        ],
    )
    parser.add_argument("--n_kernels", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "mnist"])
    parser.add_argument(
        "--channel_handling",
        type=str,
        default=None,
        choices=["separate", "concatenate", "concatenate_channelwise"],
        help="If not provided, automatically set based on model type.",
    )
    parser.add_argument(
        "--channel_method",
        type=str,
        required=True,
        choices=["zigzag", "spiral", "hilbert", "row_wise", "column_wise", "none"],
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"{timestamp}_"
        f"vrocks_{args.model}_{args.dataset}_kernels{args.n_kernels}_"
        f"{args.channel_handling}_{args.channel_method}_"
        f"seed{args.seed}_alpha{args.alpha}"
    )

    logger = Logger(
        name=run_name,
        logs_directory="./logs/",
        results_directory="./results/",
        log_metrics_directory="./log_metrics/",
    )

    config = {
        "run_name": run_name,
        "model": args.model,
        "n_kernels": args.n_kernels,
        "dataset": args.dataset,
        "channel_handling": args.channel_handling,
        "channel_method": args.channel_method,
        "alpha": args.alpha,
        "seed": args.seed,
        "n_jobs": -1,
        "classifier": "RidgeClassifier",
        #"cv": None,
        #"scaler": False,
        #"dataset_normalization": "none",
        #"gaussian_sigma": 0.0,
    }
    logger.log(json.dumps(config, indent=2), header="Configuration")

    t0 = time.time()
    t_load_start = time.time()
    X_train, y_train, X_test, y_test = load_dataset_numpy(args.dataset)
    args.channel_handling = resolve_channel_handling(args.model, args.channel_handling)
    t_load_end = time.time()

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    t_pre_start = time.time()
    X_train_pre = data_transform(X_train, args.channel_handling, args.channel_method)
    X_test_pre = data_transform(X_test, args.channel_handling, args.channel_method)
    t_pre_end = time.time()

    t_feat_start = time.time()
    rocket = build_rocket(args.model, args.n_kernels, args.seed)
    rocket.fit(X_train_pre)
    X_train_features = rocket.transform(X_train_pre)
    X_test_features = rocket.transform(X_test_pre)
    t_feat_end = time.time()

    t_clf_start = time.time()
    clf = RidgeClassifier(alpha=args.alpha)
    clf.fit(X_train_features, y_train)
    t_clf_end = time.time()

    t_eval_start = time.time()
    acc_train = accuracy_score(y_train, clf.predict(X_train_features))
    acc_test = accuracy_score(y_test, clf.predict(X_test_features))
    t_eval_end = time.time()

    t1 = time.time()
    stats = {
        "train_acc": float(acc_train),
        "test_acc": float(acc_test),
        "t_load_s": t_load_end - t_load_start,
        "t_pre_s": t_pre_end - t_pre_start,
        "t_feat_s": t_feat_end - t_feat_start,
        "t_clf_s": t_clf_end - t_clf_start,
        "t_eval_s": t_eval_end - t_eval_start,
        "t_total_s": t1 - t0,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_features_shape": list(getattr(X_train_features, "shape", ())),
        "test_features_shape": list(getattr(X_test_features, "shape", ())),
    }
    logger.log(json.dumps(stats, indent=2), header="Results", color="green")

    """  os.makedirs(logger.results_directory, exist_ok=True)
    results_path = os.path.join(logger.results_directory, f"{run_name}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "results": stats}, f, indent=2) """

    logger.finish()

if __name__ == "__main__":
    main()