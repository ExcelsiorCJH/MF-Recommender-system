import os
import argparse
import numpy as np

from model import PMF, BiasPMF, AddPMF
from utils import rmse, to_csv
from utils import get_data, train_test_split, get_rating_dict


# Paser
parser = argparse.ArgumentParser(description="Recommendation usint Matrix Factorization")
# model
parser.add_argument("--model", type=str, default="AddPMF", help="Model name to use")
parser.add_argument("--epoch", type=int, default=20, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=100000, help="Size of batches")
parser.add_argument("--features", type=int, default=100, help="Number of latent features")
parser.add_argument("--reg", type=float, default=0.04, help="Regularization parameter")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate parameter")
parser.add_argument("--alpha", type=float, default=0.4, help="Weight for user/item rating")
# data
parser.add_argument("--data_type", type=str, default="DS2", help="Dataset name to analyze")
parser.add_argument(
    "--data_path", type=str, default="data/ratings.csv", help="Folder for existing data"
)
# test
parser.add_argument(
    "--test", type=bool, default=False, help="Use model to inference without training"
)
# parser.add_argument("--load_dir", type=str, default="outputs/DS2/weights")
args = parser.parse_args()

# set seed
np.random.seed(42)


def train():

    # check directories if not make directories
    output_path = "./outputs"
    output_type_path = f"{output_path}/{args.data_type}"
    otuput_result_path = f"{output_type_path}/results"
    output_weight_path = f"{output_type_path}/weights"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_type_path):
        os.mkdir(output_type_path)
    if not os.path.exists(otuput_result_path):
        os.mkdir(otuput_result_path)
    if not os.path.exists(output_weight_path):
        os.mkdir(output_weight_path)

    # Dataset Load
    print("===== Dataset Load... =====")
    dataset = get_data(args.data_path)
    if args.data_type == "DS1":
        trainset, testset = train_test_split(dataset, data_type="DS1")
    else:
        trainset, testset = train_test_split(dataset, data_type="DS2")

    # Model Load
    n_user = int(max(np.amax(trainset[:, 0]), np.amax(testset[:, 0]))) + 1
    n_item = int(max(np.amax(trainset[:, 1]), np.amax(testset[:, 1]))) + 1

    print(f"===== {args.model} Model Load... =====")
    if args.model == "AddPMF":
        user_rating_dict, item_rating_dict = get_rating_dict(trainset)
        model = AddPMF(
            n_user=n_user,
            n_item=n_item,
            user_rating_dict=user_rating_dict,
            item_rating_dict=item_rating_dict,
            batch_size=args.batch_size,
            n_feature=args.features,
            reg=args.reg,
            alpha=args.alpha,
            momentum=args.momentum,
            learning_rate=args.lr,
            save_dir=f"outputs/{args.data_type}",
        )
    elif args.model == "BiasPMF":
        model = BiasPMF(
            n_user=n_user,
            n_item=n_item,
            batch_size=args.batch_size,
            n_feature=args.features,
            reg=args.reg,
            momentum=args.momentum,
            learning_rate=args.lr,
            save_dir=f"outputs/{args.data_type}",
        )
    elif args.model == "PMF":
        model = PMF(
            n_user=n_user,
            n_item=n_item,
            batch_size=args.batch_size,
            n_feature=args.features,
            reg=args.reg,
            momentum=args.momentum,
            learning_rate=args.lr,
            save_dir=f"outputs/{args.data_type}",
        )

    # train
    model.fit(trainset[:, :-1], validset=testset, epochs=args.epoch)

    return None


def test():
    # Dataset Load
    print("===== Dataset Load... =====")
    dataset = get_data(args.data_path)
    if args.data_type == "DS1":
        trainset, testset = train_test_split(dataset, data_type="DS1")
    else:
        trainset, testset = train_test_split(dataset, data_type="DS2")

    n_user = int(max(np.amax(trainset[:, 0]), np.amax(testset[:, 0]))) + 1
    n_item = int(max(np.amax(trainset[:, 1]), np.amax(testset[:, 1]))) + 1

    # Model Load
    print(f"===== {args.model} Model Load... =====")
    if args.model == "AddPMF":
        user_rating_dict, item_rating_dict = get_rating_dict(trainset)
        model = AddPMF(
            n_user=n_user,
            n_item=n_item,
            user_rating_dict=user_rating_dict,
            item_rating_dict=item_rating_dict,
            batch_size=args.batch_size,
            n_feature=args.features,
            reg=args.reg,
            momentum=args.momentum,
            learning_rate=args.lr,
            save_dir=f"outputs/{args.data_type}",
        )
        weights_path = f"outputs/{args.data_type}/weights/best_model_addpmf.npz"
    elif args.model == "BiasPMF":
        model = BiasPMF(
            n_user=n_user,
            n_item=n_item,
            batch_size=args.batch_size,
            n_feature=args.features,
            reg=args.reg,
            momentum=args.momentum,
            learning_rate=args.lr,
            save_dir=f"outputs/{args.data_type}",
        )
        weights_path = f"outputs/{args.data_type}/weights/best_model_bpmf.npz"
    elif args.model == "PMF":
        model = PMF(
            n_user=n_user,
            n_item=n_item,
            batch_size=args.batch_size,
            n_feature=args.features,
            reg=args.reg,
            momentum=args.momentum,
            learning_rate=args.lr,
            save_dir=f"outputs/{args.data_type}",
        )
        weights_path = f"outputs/{args.data_type}/weights/best_model_pmf.npz"

    # load pret-trained weights
    model.load_weights(weights_path)

    # test
    print(f"===== {args.data_type} Inference... =====")
    preds = model.predict(testset[:, :2])
    test_loss = rmse(preds, testset[:, 2])
    print(f"test RMSE: {test_loss}")
    to_csv(f"./B_results_{args.data_type}.csv", preds, testset, header=True)

    return None


if __name__ == "__main__":
    if args.test:
        print(f"******** {args.model} / {args.data_type} Test Start ********")
        test()
    else:
        print(f"******** {args.model} / {args.data_type} Train Start ********")
        train()

