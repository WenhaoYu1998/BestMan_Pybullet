# Installation of anygrasp

> &emsp;You need to get anygrasp [license and checkpoint](./README.md) to use it.


- Python 3.10

```
conda create -n anygrasp -y python=3.10
conda activate anygrasp
```

> &emsp;You need `export MAX_JOBS=2` in terminal before pip install if you are running on an laptop due to [this issue](https://github.com/NVIDIA/MinkowskiEngine/issues/228).

- Install MinkowskiEngine
  
```
# Prerequisites
conda install openblas-devel -c anaconda -y
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install nvidia/label/cuda-11.6.0::libcusolver-dev -y
conda install nvidia/label/cuda-11.6.0::libcurand-dev -y 
conda install ninja -y

# Install
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --global-option=build_ext --global-option="--blas_include_dirs=${CONDA_PREFIX}/include" --global-option="--blas=openblas"


```

- Install other requirements of anygrasp

> &emsp; cd to anygrasp project

```
# Due to graspnetAPI specifies the use of sklearn, allow to install deprecated sklearn package
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install -r requirements.txt
```

- Install ``pointnet2`` module.
```bash
    cd pointnet2
    python setup.py install
```

- Some package version change and install
```
pip install numpy==1.23.5 --force-reinstall
pip install yacs
```

- Other setting
#You may encounter the following problems about open3d show:

```
.../libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by ...)
```

> &emsp; Establish symbolic link to solve this problem

```
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 {anaconda3_path}/envs/anygrasp/lib/python3.10/site-packages/torch/lib/../../../../libstdc++.so.6
```

change {anaconda3_path} to you anaconda3 install path