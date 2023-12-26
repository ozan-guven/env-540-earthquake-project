#!/bin/bash

(
    cd ../src/sweepers
    #(conda run -n few --no-capture-output --live-stream python unet_sweeper.py)
    #(conda run -n few --no-capture-output --live-stream python siamese_unet_conc_sweeper.py)
    (conda run -n few --no-capture-output --live-stream python siamese_unet_diff_sweeper.py)
)