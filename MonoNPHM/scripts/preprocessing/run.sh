#!/bin/bash
python run_PIPNet.py --seq_name $1
python run_facer.py --seq_name $1
#python run_matting_images.py --seq_name $1
python run_MICA.py --seq_name $1
python run_metrical_tracker.py --seq_name $1 --no-intrinsics_provided
python run_normal_predictor.py --seq_name $1
