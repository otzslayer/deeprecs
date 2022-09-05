ncf_args = {
    "train_rating": "../NCF/ml-1m.train.rating",
    "test_negative": "../NCF/ml-1m.test.negative",
    "epochs": 10,
    "batch_size": 256,
    "top_k": 10,
    "num_ng": 4,
    "test_num_ng": 99,
    "model_name": "NCF",
    "model_save": True,
    "model_path": "../results /",
    "device": "cpu",
}
