# examples

### generate relations from a text file

```shell
python -m run.generate_relation_dataset --head 5 --outpath ~/data/gg/experiments/cheops --input-txt ./test/data/cheops.txt
```


## CUDA notes

current setup works with CUDA 11.4, which can be installed using cuda_11.4.4_470.82.01_linux.run

E.g. wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run

```shell
apt install nvidia-driver-510
```

NB: if you encounter failures due to some `nvidia.uvm`. You may want to reboot, after wiping the previous cuda. Then you might want to unload nouveau driver, as described here : https://askubuntu.com/questions/841876/how-to-disable-nouveau-kernel-driver  
