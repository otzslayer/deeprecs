import os

import torch
from torch.utils import data

from deeprecs.data.ncf_data import NCFData, load_data
from deeprecs.models.ncf import NCF
from deeprecs.utils.ncf_args import ncf_args

# set args

PATH_TRAIN_DATA = str(ncf_args["train_rating"])
PATH_TEST_DATA = str(ncf_args["test_negative"])
epochs = int(ncf_args["epoch"])
batch_size = int(ncf_args["batch_size"])
top_k = int(ncf_args["top_k"])
num_ng = int(ncf_args["num_ng"])
test_num_ng = int(ncf_args["test_num_ng"])
model_name = str(ncf_args["model_name"])
model_save = bool(ncf_args["model_save"])
model_path = str(ncf_args["model_path"])
device = str(ncf_args["device"])

train_data, test_data, num_user, num_item, train_mat = load_data(
    PATH_TRAIN_DATA, PATH_TEST_DATA
)

train_dataset = NCFData(
    train_data, num_item, train_mat, num_ng=num_ng, is_training=True
)

test_dataset = NCFData(
    test_data, num_item, train_mat, num_ng=0, is_training=False
)

train_loader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)

test_loader = data.DataLoader(
    test_dataset, batch_size=test_num_ng + 1, shuffle=False, num_workers=0
)

# train/eval model
model = NCF(
    num_user,
    num_item,
    model=model_name,
    num_factor=8,
    layers=[64, 32, 16, 8],
    device="cpu",
)

for epoch in range(epochs):
    loss = model.fit(train_loader=train_loader)

    hr, ndcg = model.evaluate(test_loader=test_loader, top_k=10)

    print(f"Epoch {epoch} : HR = {hr}, NDCG = {ndcg}, Loss = {loss}")

    best_hr = 0
    if hr > best_hr:
        best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
        if model_save:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model, f"{model_path}{model_name}_epoch{epoch}.pth")

    print(f"End. Best epoch {best_epoch} : HR = {best_hr}, NDCG = {best_ndcg}")
