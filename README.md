## 📄 About This Repository

This is the official repository for the paper:

**"One-Shot Real-to-Sim via End-to-End Differentiable Simulation and Rendering"**  
Published in *IEEE Robotics and Automation Letters (RA-L), 2025*

**Authors**: Yifan Zhu, Tianyi Xiang, Aaron M. Dollar, and Zherong Pan

## Installation Instructions
This repo also depends on our fork of the original Shape As Points repo. Please clone it (https://github.com/yifanzhu95/shape_as_points.git) in the diffworld/shape_as_points folder. [TODO: add this as a submodule.]

The code is tested on Ubuntu 22.04.5 LTS, with Python3.9 and Cuda 12.1. We tested on a single RTX 4090 GPU. We recommend using a GPU with 12GB or more of VRAM. We also recommend using a machine with at least 16GB of RAM. To start with, make sure you already have Cuda 12.1 installed, then:

Create a new conda environment and activate it:
```
conda create -n rigidworldmodel python=3.9
conda activate rigidworldmodel
```

First install PyTorch:
```
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 -f https://download.pytorch.org/whl/torch_stable.html
```

To install the other dependencies:
```
pip install -r requirements.txt
```



## Running the Code
We provide the code and data for running the stage 1 and stage 2 training for the Drill object in the simulation experiments. Download the data [here](https://www.dropbox.com/scl/fi/jb030a8nu4ahoie9zs5iq/drill.tar.gz?rlkey=kqtbedwmlz9gcrrkisp6kghmf&st=88lammvr&dl=0) and place it in the /data folder in the project root folder (create this folder if it doesn't exist). Note that you can place the data anywhere you want, but you will need to change the data path in the configuration file (e.g. the "stage_1_data_folder" entry in the configs/drill_stage_1.yaml file).


### The structure of the repo should be: 
```
RigidWorldModel/ <---- project root
|-asset/ <---- folder for storing initial object geometry mesh
|-checkpoints/ <---- folder for checkpoints (will be created during the first training run)
|-diffworld/ <---- source code
|  |-data/
|  |-diffsdfsim/
|  |-shape+as_points/
|  |-utils/
|  |-world/
|-data/ <---- training data
|  |-drill/
|  |  |train1/
```

### Training 
To run stage 1 training:
```
python main_stage_1.py --config=configs/drill_stage_1.yaml
```

The trained model and other debugging output will be saved in the /checkpoints/drill1 folder. We highly recommend that you use Weights&Biases for logging and visualization. You can run the code with the following command:
```
python main_stage_1.py --config=configs/drill_stage_1.yaml --use_wandb
```

Once training is done, you can run stage 2 (assuming you are uisng Weights&Biases):
```
python main_stage_2.py --config=configs/drill_stage_2.yaml --use_wandb
```
The trained model and other debugging output will be saved in the /checkpoints/drill2 folder.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.