# VEHICLE SEGMENTATION IN MLS POINT CLOUD

The goal is to accurately Identify the vehicle from occluded and incomplete point cloud form mobile laser scanning (MLS) point cloud data.

## How its different form Yanx27 PointNet++ repository
I simplified the process of data input path, added steps/hints inside the code to learn easily each step, modified the repository for binary segmentation ["Vehicle", "Non-vehicle"], and improved the time required to test the model.

## INITIAL STEPS

Process can be done in Google colab, Ubuntu or windows. All the step has been done and tested in all of the platform.
I used Conda environment to set-up the process. Download the Anaconda package from [here](https://www.anaconda.com/download/). After Installing create a new environment named pointnet. 

```bash
conda create -n pointnet python
conda activate pointnet
```
Install the dependency using the following command.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### DATA FORMAT STRUCTURE
d
``` git


```


### CLONE THE REPOSITORY
In Ubuntu/Windows
``` git
git clone https://github.com/bhagatdas/Vehicle-segmentation-in-MLS-point-cloud.git
```

In google colab
``` git
!git clone 'https://github.com/bhagatdas/Vehicle-segmentation-in-MLS-point-cloud.git' '/content/drive/My Drive/pointnetpp'
```

## RUN
Preprocessing:

```python
cd data_utils
python collect_indoor3d_data.py
```
Training:

```python
cd ..
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg --optimizer Adam --epoch 32
```

Testing:
```python
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
