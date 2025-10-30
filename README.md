# Comparative Analysis of Brain MRI Data Using DeepPrep on FABRIC


### 📘 Overview

This project demonstrates the use of **DeepPrep**, a deep-learning-powered neuroimaging preprocessing pipeline, to perform a **comparative analysis of brain MRI (sMRI and fMRI)** data on the **FABRIC research testbed**.  

DeepPrep provides an automated preprocessing framework that integrates widely used neuroimaging tools such as **FSL**, **FreeSurfer**, and **AFNI**, all packaged within a Docker container for reproducibility and scalability.

This guide walks through environment setup, data preparation, Docker configuration, and execution of DeepPrep on a GPU-enabled FABRIC node.


### ⚙️ System Requirements

| Resource | Recommended Configuration |
|-----------|----------------------------|
| **CPU** | 8 cores |
| **Memory (RAM)** | 16 GB |
| **GPU** | Any NVIDIA GPU (CUDA-compatible) |
| **Storage** | Minimum 50 GB free disk space |
| **Operating System** | Ubuntu 20.04 or higher |
| **Internet Access** | Required for dataset and Docker image download |


### 🧩 Prerequisites

Before beginning, ensure that you can:

1. Access the **FABRIC Research Testbed**.
2. Provision and SSH into a **virtual node**.
3. Have administrative (sudo) privileges on your node.


### 🖥️ Step 1: Create a Virtual Node on FABRIC

1. Log in to your **FABRIC Portal** and create a **single-node machine** with:
   - 8 vCPUs  
   - 16 GB RAM  
   - 1 NVIDIA GPU (e.g., A100, V100, or L40S)
2. Once provisioned, SSH into your node:
   ```bash
   ssh ubuntu@<your_node_ip>
3. Verify your hardware configuration:
 ```  
lscpu          # check CPU details
free -h        # check memory allocation
nvidia-smi     # check GPU availability
```

### 🧱 Step 2: Install Dependencies
Install Docker, NVIDIA Container Toolkit, and CUDA Drivers.

#### Update the system
```
sudo apt update && sudo apt upgrade -y
```
#### Install Docker
```
sudo apt install -y docker.io
```

#### Enable and start Docker
```
sudo systemctl enable docker
sudo systemctl start docker
```

#### (Optional) Add current user to Docker group
```
sudo usermod -aG docker $USER
newgrp docker
```
#### Verify Docker installation
```
docker --version
```
#### Install NVIDIA driver and container toolkit
```
sudo apt install -y nvidia-driver-550 nvidia-container-toolkit
sudo systemctl restart docker
```

#### Verify GPU access from inside Docker
```
sudo docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

### 🧠 Step 3: Download the Dataset
Download the dataset from Kaggle:
Dataset Link: [Professional Chess Players sMRI and fMRI Dataset](https://www.kaggle.com/datasets/ajay2529kumar/professional-chess-players-smri-and-fmri-dataset) Once downloaded, extract the data into a folder named `chess_data` in your home directory:

```
mkdir -p ~/chess_data
cd ~/chess_data
# Move or extract dataset here


~/chess_data/bids/
├── sub-01/
│   ├── anat/
│   └── func/
├── sub-02/
│   ├── anat/
│   └── func/
└── dataset_description.json
```

### 🧾 Step 4: Obtain and Set Up FreeSurfer License
DeepPrep uses FreeSurfer for anatomical preprocessing.
You need a valid FreeSurfer license.
Register for a free license at: 🔗 https://surfer.nmr.mgh.harvard.edu/registration.html
After receiving license.txt via email, place it under:

```
~/freesurfer_key/license.txt
```

### 🐋 Step 5: Pull the DeepPrep Docker Image
Download the official DeepPrep container from DockerHub:
```
sudo docker pull pbfslab/deepprep:25.1.0

#Verify the image is downloaded:
sudo docker images | grep deepprep
```

### 🚀 Step 6: Run DeepPrep
Use the command below to execute DeepPrep on your dataset.
```
sudo docker run -it --rm --gpus all \
  -v ~/chess_data/bids:/input \
  -v ~/output:/output \
  -v ~/freesurfer_key/license.txt:/fs_license.txt \
  pbfslab/deepprep:25.1.0 \
  /input /output participant \
  --fs_license_file /fs_license.txt \
  --bold_task_type rest \
  --cpus 8 \
  --memory 16
```

Explanation of Command Parameters:
Parameter	Description
```--gpus all	Enables GPU acceleration inside Docker
-v ~/chess_data/bids:/input	Mounts the input dataset
-v ~/output:/output	Mounts the output directory
-v ~/freesurfer_key/license.txt:/fs_license.txt	Mounts the FreeSurfer license
/input /output participant	Specifies BIDS input/output folders
--bold_task_type rest	Processes resting-state fMRI data
--cpus 8, --memory 16	Allocates resources for processing
```

### 📂 Step 7: Verify Output
Once DeepPrep completes, preprocessed outputs will appear in:
```
~/output/
tree ~/output | less
~/output/
├── sub-01/
│   ├── func/
│   ├── anat/
│   └── figures/
├── reports/
└── logs/
```

### 📊 Step 8: Monitor System Usage
During or after execution, you can check hardware utilization:
```
top
htop
nvidia-smi
```

### 🛠️ Troubleshooting
Issue	Possible Fix
| Issue                          | Possible Fix                                                                                   |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| **CUDA not found**             | Ensure correct NVIDIA drivers and toolkit are installed. Test with `nvidia-smi` inside Docker. |
| **Permission denied (Docker)** | Add your user to the Docker group: `sudo usermod -aG docker $USER` then `newgrp docker`.       |
| **No output generated**        | Confirm input data follows the BIDS structure and check logs under `~/output/logs/`.           |
| **Memory errors**              | Increase virtual node memory or reduce the number of CPUs in the command.                      |


### 📚 Citation  
If you use this project, dataset, or pipeline, please cite the following:
DeepPrep: A Deep Learning Enabled Neuroimaging Preprocessing Framework PBFSLab, 2025.
Docker Image: pbfslab/deepprep:25.1.0
