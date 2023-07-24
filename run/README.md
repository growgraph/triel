# examples

### generate relations from a text file

```shell
python -m run.generate_relation_dataset --head 5 --outpath ~/data/gg/experiments/cheops --input-txt ./test/data/cheops.txt
```


## CUDA notes

nvidia version 510 : 
```shell
apt install nvidia-driver-510
```

cuda version 11.6 : 

```shell
wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
sh cuda_11.6.2_510.47.03_linux.run
```

spacy cuda 116 : `spacy = {extras = ["cuda116"], version = "3.5"}`