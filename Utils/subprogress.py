import os
import pickle
import subprocess

import numpy as np
from PIL import Image
from yacs.config import CfgNode as CN


class Submodule:

    def __init__(self):
        self.data_dict = {}

    def transform(self, data, type=None):
        if type is None:
            if isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, CN):
                return pickle.dumps(data)
            elif isinstance(data, Image.Image):
                return pickle.dumps(data)
            else:
                return data
        else:
            if type == np.ndarray:
                return np.array(data)
            elif type == CN:
                return pickle.loads(data)
            elif type == Image.Image:
                return pickle.loads(data)
            else:
                return data

    def add(self, name, data):
        self.data_dict[name] = self.transform(data)

    def get(self, name, type=None):
        return self.transform(self.data_dict[name], type)

    def clear(self):
        self.data_dict.clear()

    def set_pkl(self, pkl_file):
        self.pkl_file = pkl_file

    def serialize(self, pkl_file):
        with open(pkl_file, "wb") as f:
            pickle.dump(self.data_dict, f)

    def deserialize(self, pkl_file):
        with open(pkl_file, "rb") as f:
            self.data_dict = pickle.load(f)

    def call(self, env_name, script_path):
        print(
            f"[Submodule call] \033[34mInfo\033[0m: Start run submodule {script_path} in {env_name}"
        )
        script_path = os.path.join("..", script_path)
        pkl_file = os.path.join(
            os.path.dirname(os.path.abspath(script_path)), "data.pkl"
        )
        self.serialize(pkl_file)
        result = subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                env_name,
                "python",
                script_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"[Submodule call] \033[31merror\033[0m: Error occurred: {result.stderr}"
            )
        print(result.stdout)
        self.deserialize(pkl_file)
        print(
            f"[Submodule call] \033[34mInfo\033[0m: End run submodule {script_path} in {env_name}!"
        )
