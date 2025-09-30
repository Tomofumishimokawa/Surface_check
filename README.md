# Surface_check
import os
os.environ["DDE_BACKEND"] = "pytorch"  # ← これを deepxde より先に設定
import deepxde as dde
import torch
import deepxde.backend as bkd
print("Current DeepXDE backend:", bkd.backend_name)
