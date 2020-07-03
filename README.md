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
    ├── train_img                   # n frame time-lapse images
    │   ├── 0000.png                # Image name with '0' characters padded to the left
    │   ├── 0001.png
    │   ├── :.png
    │   ├── n-1.png
    │   └── n.png
    └── train_mpm
    │   ├── 001                     # Each frame-interval of MPM
    │   │   ├── 0000.npy            # A.npy is MPM between "frame A" and "frame A + interval"
    │   │   ├── 0001.npy
    │   │   ├── 0002.npy
    │   │   ├── :.npy
    │   │   ├── n-2.npy
    │   │   └── n-1.npy
    │   └── m
    │       ├── 0000.npy
    │       ├── 0001.npy
    │       ├── :.npy
    │       ├── n-m-1.npy
    │       └── n-m.npy
    ├── eval_img
    │       (Same structure of train_img. Without eval_img, part of train_img is used for evaluation)
    ├── eval_mpm
             (Same structure of train_mpm.)
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
