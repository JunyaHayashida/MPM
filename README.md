# MPM: Motion and Position Map 
## Genarate sample MPMs
arguments:
    1. tracklet path
    2. sample image path
    3. output directory path
```
$ python3 mpm_genarator.py sample/sample_tracklet.txt sample/sample_img/000.png sample/output --intervals 1,3,5
```

## Train MPMs
arguments:
    1. epochs
    2. batch size
    3. dataset directory path
```
$ python3 mpm_train.py 100 40 sample/train
```
coming soon

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
