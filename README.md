# labellio_classify
labellio_classifyは、labellioからダウンロードしたモデルを評価するためのツールです。  
以下のモジュールがインストールされていることが前提となります。
```sh
python 2.7
tensorflow
imghdr
numpy
```

あらかじめ、識別したい画像をひとつのディレクトリ内に保存しておきます。		
```sh
$ ls images
dog1.jpg  dog2.jpg  cat1.jpg  cat2.jpg
```

ダウンロードしたモデルを展開します。
```sh
$ ls model-(iteration数)
〇〇-(iteration数).ckpt.data00~ 〇〇-(iteration数).ckpt.meta 〇〇-(iteration数).ckpt.index label.txt checkpoint
```

以下のように、モデルのディレクトリ名・画像のディレクトリ名・モデルのネットワーク名(mobilenet_v1、inception_v4 もしくは resnet_v2_152)を指定し、classify.pyを起動します。		
```sh
$ python classify.py model_directory image_directory network_name
...
images/dog2.jpg	cat	[ 0.02713127  0.97286868]
images/dog1.jpg	cat	[  7.87437428e-04   9.99212503e-01]
images/cat1.jpg	dog	[ 0.99114448  0.00885548]
images/cat2.jpg	cat	[ 0.44422832  0.55577165]
```
