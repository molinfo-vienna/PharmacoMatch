import os


def load_model_from_path(path, model):
    for file in os.listdir(path):
        if file.endswith(".ckpt"):
            path = os.path.join(path, file)
    return model.load_from_checkpoint(path)
