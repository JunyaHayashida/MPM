# MPM: Motion and Position Map 
## Requirements

- Python3.6
- PyTorch
- OpenCV
- Hydra (hydra-core)
## Genarate sample MPMs
### Arguments
You can set up input path/output path/parameters from 
[config/mpm_generator.yaml](https://github.com/JunyaHayashida/MPM/blob/master/config/mpm_generator.yaml)
### Example
```
$ python3 mpm_genarator.py
```

## Train MPMs
### Preparation
Please prepare your data as follows

<details><summary>current dir</summary><div>

```
./data
    ├── train_imgs
    │   ├── sequenceA                      # Arbitrary sequence name
    │   │   ├── 0000.png                   # Frame index with '0' characters padded to the left
    │   │   ├── 0001.png
    │   │   ├── :
    │   │   └── number of frames -1.png
    │   └── sequenceB, C, D, ...           # In case of using 2 or more sequences
    │       ├── 0000.png
    │       ├── 0001.png
    │       ├── :
    │       └── number of frames -1.png
    ├── train_mpms
    │   ├── sequenceA                      # Sequence name on train_imgs/*
    │   │   ├── n                          # Arbitrary frame-interval of MPM with '0' characters padded to the left
    │   │   │   ├── 0000.npy               # n/0000.npy is MPM between "frame 0" and "frame 0 + n"
    │   │   │   ├── 0001.npy
    │   │   │   ├── :
    │   │   │   └── number of frames -1 -n.npy
    │   │   └── m                          # In case of using 2 or more frame-intervals of MPM
    │   │       ├── 0000.npy               # m/0000.npy is MPM between "frame 0" and "frame 0 + m"
    │   │       ├── 0001.npy
    │   │       ├── :
    │   │       └── number of frames -1 -m.npy
    │   └── sequenceB, C, D, ...
    │               (Same structure of train_mpms/sequenceA/*)
    ├── eval_imgs
    │       (Same structure of train_imgs. Without eval_img, part of train_img is used for evaluation)
    └── eval_mpms
            (Same structure of train_mpms)
```
</div></details>

### Arguments
You can set up input path/output path/parameters from 
[config/mpm_train.yaml](https://github.com/JunyaHayashida/MPM/blob/master/config/mpm_train.yaml)

### Example   
```
$ python3 mpm_train.py
```
## Track cells
coming soon

## Citation
If you find the code useful for your research, please cite:
```
@inproceedings{Hayashida2020MPM,
  author = {Hayashida, Junya and Nishimura, Kazuya and Bise, Ryoma}
  title = {MPM: Joint Representation of Motion and Position Map for Cell Tracking},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
