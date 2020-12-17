nohup python object_detection/train.py \
        --logtostderr \
        --train_dir=train \
        --pipeline_config_path=configs\faster_rcnn_resnet101.config > train/losslog/train_loss.log 2>&1 &
