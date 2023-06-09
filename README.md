# VEHICLE SEGMENTATION IN MLS POINT CLOUD

The goal of this project is to accurately identify vehicles from occluded and incomplete point cloud data obtained through mobile laser scanning (MLS) using the PointNet, PointNet++, and PointNet++ MSG models.

![pointnet++](https://github.com/bhagatdas/Vehicle-segmentation-in-MLS-point-cloud/blob/master/pointnetpp.tif)

## THE CHANGES I MADE FROM [YANX27](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

In this forked repository, I have made several changes and improvements based on the original repository by YANX27:

1. Simplified the process of specifying the data input path.
2. Added detailed steps and hints within the code to facilitate easier understanding of each step.
3. Modified the repository to enable binary segmentation of vehicles and non-vehicles.
4. Improved the testing time required for the models.



## INITIAL STEPS

The following steps outline the initial setup process. This project can be executed on Google Colab, Ubuntu, or Windows. The steps provided below are specifically for the Ubuntu environment. Code related to Google Colab can be found in the ***run_colab.ipynb*** notebook. 
Conda environment was used for the setup process.
1. Download the Anaconda package from here and install it.
2. After installation, create a new environment named 
pointnet by executing the following commands in the terminal:

```bash
conda create -n pointnet python
conda activate pointnet
```

3. Install the necessary dependencies by running the following commands:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

4. To obtain the code for this project, clone the repository by executing the following command:
``` git
git clone https://github.com/bhagatdas/Vehicle-segmentation-in-MLS-point-cloud.git
```

## DATA FOLDER STRUCTURE
- input (root) 
  - Area_1
    - Section_1
      - Annotations
        - vehicle.txt
        - non-vehicle.txt
  - Area_2
    - Section_1
      - Annotations
        - vehicle.txt
        - non-vehicle.txt




## RUN
Perform the following steps to run the code:

1. Preprocessing:

```python
cd data_utils
python collect_indoor3d_data.py
```
2. Training:

```python
cd ..
python train_semseg.py --model pointnet2_sem_seg --test_area 2 --log_dir pointnet2_sem_seg --optimizer Adam --epoch 32
```

3. Testing:
```python
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 2 --visual
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
