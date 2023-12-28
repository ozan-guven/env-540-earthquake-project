#!/bin/bash

(
    cd ../src/contrastive
    (conda run -n few --no-capture-output --live-stream python contrastive_unet_train.py)
    (conda run -n few --no-capture-output --live-stream python contrastive_siamese_unet_train.py)
    cd ../unet
    (conda run -n few --no-capture-output --live-stream python unet_train.py)
    (conda run -n few --no-capture-output --live-stream python unet_train.py --use_pretrained)
    (conda run -n few --no-capture-output --live-stream python unet_train.py --use_pretrained --freeze_encoder)
    cd ../siamese_unet_conc
    (conda run -n few --no-capture-output --live-stream python siamese_unet_conc_train.py)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_conc_train.py --use_pretrained)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_conc_train.py --use_pretrained --freeze_encoder)
    cd ../siamese_unet_diff
    (conda run -n few --no-capture-output --live-stream python siamese_unet_diff_train.py)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_diff_train.py --use_pretrained)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_diff_train.py --use_pretrained --freeze_encoder)
)