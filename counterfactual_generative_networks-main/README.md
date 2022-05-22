Training an invariant ImageNet classifier:

The main functions of this sub-repo are:

- Training a CGN
- Generating data (samples, interpolations, or a whole dataset)
- Training an invariant classifier ensemble



### Train the Invariant Classifier Ensemble ###

__Training__. First, you need to make sure that you have all datasets in ```imagenet/data/```. Download mini_Imagenet from [Kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000). You also need produce a counterfactual dataset by CGN code. Finally, style transfer them all.  Cue-Conflict needs be downloading via the download script in ```scripts```.

The datasets we made can be downloaded from [google drive](https://drive.google.com/drive/folders/1Gb7P2tHKLMpHPIimxq0hn5G-608wiRiI?usp=sharing) and unzipped. So the data folder should look like:

    /data
        /cgn
        /cgn_style
        /cue_conflict
        /imagenet_mini
        /imagenet_style

To train a classifier on a single GPU with a pre-trained Resnet-50 backbone, run

```Shell
python imagenet/train_classifier.py  -j 3 \
--epochs 45 --pretrained --cf_data CF_DATA_PATH --name RUN_NAME\
--data imagenet/data/imagenet_mini --cf_data imagenet/data --cf_style_data imagenet/data/cgn_style --style_training=True --imagenet_training=True --cf_training=True --cf_style_training=True --name img_style-img_cgn_style-cgn
```

`--style_training`: if to use style-ImageNet for training

`--imagenet_training` : if to use ImageNet for training

`--cf_training`: if to use cgn-ImageNet for training

`--cf_style_training`: if to use cgn-style-ImageNet for training



Again, add ```--help``` for more information on the possible arguments.