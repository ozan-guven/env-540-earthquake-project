#!/bin/bash

(
    cd ../src/unet
    (conda run -n few --no-capture-output --live-stream python unet_test.py)
    (conda run -n few --no-capture-output --live-stream python unet_test.py --use_pretrained)
    (conda run -n few --no-capture-output --live-stream python unet_test.py --use_pretrained --freeze_encoder)
    cd ../siamese_unet_conc
    (conda run -n few --no-capture-output --live-stream python siamese_unet_conc_test.py)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_conc_test.py --use_pretrained)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_conc_test.py --use_pretrained --freeze_encoder)
    cd ../siamese_unet_diff
    (conda run -n few --no-capture-output --live-stream python siamese_unet_diff_test.py)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_diff_test.py --use_pretrained)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_diff_test.py --use_pretrained --freeze_encoder)
)