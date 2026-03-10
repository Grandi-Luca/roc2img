import os
import random
import time
import argparse
import json
from logger import Logger
from naive_r2i_functions import load_dataset_numpy, data_transform
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.rocket import (Rocket, MiniRocket, MiniRocketMultivariate)
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def set_random_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="naive_r2i baseline (ROCKET features + Ridge classifier)")
    parser.add_argument("--n_kernels", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "mnist", "adni"])
    parser.add_argument(
        "--channel_handling",
        type=str,
        default="separate",
        choices=["separate", "concatenate", "concatenate_channelwise"],
    )
    parser.add_argument(
        "--channel_method",
        type=str,
        required=True,
        choices=["zigzag", "spiral", "hilbert", "row_wise", "column_wise", "none"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaler", type=str, default="false", choices=["true", "false"])
    args = parser.parse_args()

    set_random_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"rocket{timestamp}_"
        f"naive_r2i_{args.dataset}_kernels{args.n_kernels}_"
        f"{args.channel_handling}_{args.channel_method}_"
        f"seed{args.seed}_alpha0.1"
    )

    logger = Logger(
        name=run_name,
        logs_directory="./logs/",
        results_directory="./results/",
        log_metrics_directory="./log_metrics/",
    )

    config = {
        "run_name": run_name,
        "n_kernels": args.n_kernels,
        "dataset": args.dataset,
        "channel_handling": args.channel_handling,
        "channel_method": args.channel_method,
        "alpha": 0.1,
        "seed": args.seed,
        "n_jobs": -1,
        "classifier": "RidgeClassifier",
        "scaler": args.scaler,
        #"cv": None,
        #"scaler": False,
        #"dataset_normalization": "none",
        #"gaussian_sigma": 0.0,
    }
    logger.log(json.dumps(config, indent=2), header="Configuration")

    t0 = time.time()
    t_load_start = time.time()
    X_train, y_train, X_test, y_test = load_dataset_numpy(args.dataset)
    t_load_end = time.time()

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    

    t_pre_start = time.time()
    X_train_pre = data_transform(X_train, args.channel_handling, args.channel_method)
    X_test_pre = data_transform(X_test, args.channel_handling, args.channel_method)
    t_pre_end = time.time()
    
    print("Data transformed. Shapes:")
    print("X_train_pre shape:", X_train_pre.shape)
    print("X_test_pre shape:", X_test_pre.shape)

    t_feat_start = time.time()
    rocket = Rocket(num_kernels=args.n_kernels, random_state=args.seed, n_jobs=-1)
    print(f"Channels={X_train_pre.shape[1]}")
    rocket.fit(X_train_pre)
    X_train_features = rocket.transform(X_train_pre)
    X_test_features = rocket.transform(X_test_pre)
    t_feat_end = time.time()
    
    print("X_train_features shape:", X_train_features.shape)
    # Diagnostica matrice
    print("Std minima feature:", np.std(X_train_features, axis=0).min())
    print("Condition number:", np.linalg.cond(X_train_features))
    
    if args.scaler == "true":
        scaler = StandardScaler(with_mean=False)  # Centering may not be needed for sparse features
        X_train_features = scaler.fit_transform(X_train_features)
        X_test_features = scaler.transform(X_test_features)
        print("Features scaled.")

    t_clf_start = time.time()
    clf = RidgeClassifier(alpha=0.1, random_state=args.seed)
    clf.fit(X_train_features, y_train)
    t_clf_end = time.time()
    
    print("Classifier trained.")

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