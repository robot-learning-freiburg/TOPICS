#!/usr/bin/env bash
task="configs/endovis/task_files/l2_increment_5tasks"
method="FT"
base_setting="--step 0 --epoch 60 --lr 0.02"

### PLOP 
method="PLOP"
run_setting="--sample_num 2 --dataset endovis --batch_size 24 --autocast --crop_size 512"
name="eucl_rn_v4.0_ft_sq_v2_60ep"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 4477 run.py --name ${name} --task ${task} --method ${method} ${base_setting} ${run_setting}

name="plop-v4-60ep-t5-lr0.001"
weights="--step_prev_ckpt output/endovis_l2_increment_5tasks-ov/eucl_rn_v4.0_ft_sq_v2_60ep/step0/model_final.pth"
for t in 1 2 3 4; do
 incr_setting="--step ${t} --epoch 60 --lr 0.001 --crop_size 512"
 python -m torch.distributed.launch --nproc_per_node=2 --master_port 4477 run.py --name ${name} --task ${task} --method ${method} ${incr_setting} ${weights} ${run_setting}
 weights=""
done

### EUCL BASE TRAINING for MiB and AWT

name="eucl_rn_v4.0_ft_lr2_60ep"
base_setting="--step 0 --epoch 60 --lr 0.02"
run_setting="--sample_num 2 --dataset endovis --batch_size 24 --autocast"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 4477 run.py --name ${name} --task ${task} --method ${method} ${base_setting} ${run_setting}

### MiB

method="MiB"
name="mib-v4-60ep-t5-lr0.001"
weights="--step_prev_ckpt output/endovis_l2_increment_5tasks-ov/eucl_rn_v4.0_ft_lr2_60ep/step0/model_final.pth"
for t in 1 2 3 4; do
 incr_setting="--step ${t} --epoch 60 --lr 0.01"
 python -m torch.distributed.launch --nproc_per_node=2 --master_port 4477 run.py --name ${name} --task ${task} --method ${method} ${incr_setting} ${weights} ${run_setting}
 weights=""
done

### AWT

method="AWT"
name="awt-v4-60ep-t5_lr0.001"
weights="--step_prev_ckpt output/endovis_l2_increment_5tasks-ov/eucl_rn_v4.0_ft_lr2_60ep/step0/model_final.pth"
for t in 1 2 3 4; do
 incr_setting="--step ${t} --epoch 60 --lr 0.01"
 awt_setting="--compute_att --wandb --crop_size 512"
 python run.py --name ${name} --task ${task} --method ${method} ${incr_setting} ${weights} ${run_setting} ${awt_setting}
 python -m torch.distributed.launch --nproc_per_node=2 --master_port 4477 run.py --name ${name} --task ${task} --method ${method} ${incr_setting} ${weights} ${run_setting}
 weights=""
done

### DKD

run_setting="--sample_num 2 --dataset endovis --batch_size 32 --autocast"
method="DKD"
name="dkd-v4-60ep-t5"
base_setting="--step 0 --epoch 60 --lr 0.001"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 1234 run.py --name ${name} --task ${task} --method ${method} ${base_setting} ${run_setting}
for t in 1 2 3 4; do
 incr_setting="--step ${t} --epoch 60 --lr 0.0001"
 python -m torch.distributed.launch --nproc_per_node=2 --master_port 1234 run.py --name ${name} --task ${task} --method ${method} ${incr_setting} ${run_setting}
 weights=""
done
