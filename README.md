# labellio_classify
The labellio_classify is a tool for evaluating models downloaded from labellio.  
It is assumed that the following modules are installed.
```sh
python 3.5
tensorflow 1.4.0
numpy 1.13.3
```

First, place all of the images to classify in a directory.
```sh
$ ls images
label01.jpg  label02.jpg  label03.jpg  label04.jpg label05.jpg
```

Expand the downloaded model.
```sh
$ ls model-(iteration number)
**-(iteration number).ckpt.data00~ **-(iteration number).ckpt.meta **-(iteration number).ckpt.index label.txt
```

Execute labellio_classify command providing the directory of the model, the directory of the images, and the name of the network name (mobilenet_v1, inception_v4 or resnet_v2_152).	
```sh
$ python classify.py model_directory image_directory network_name
...
images/label01.jpg	"label01":0.81,"label99":0.10,"label02":0.05,"label98":0.03,"label03":0.01	
images/label02.jpg	"label02":0.93,"label01":0.05,"label03":0.01,"label99":0.00,"label04":0.00
images/label03.jpg	"label03":0.74,"label02":0.20,"label04":0.02,"label01":0.02,"label05":0.01
images/label04.jpg	"label04":0.61,"label03":0.22,"label05":0.10,"label02":0.01,"label06":0.01
images/label05.jpg	"label05":0.56,"label06":0.7,"label04":0.05,"label07":0.03,"label03":0.03
```
