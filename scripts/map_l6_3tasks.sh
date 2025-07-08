task="configs/map/task_files/l6_increment_3tasks"
method="TOPICS"

name="hyp_rn_v4.0_hiera_mh3_ep60_c2"
run_setting="--sample_num 2 --dataset map --batch_size 24"
base_setting="--step 0 --epoch 60 --lr 0.05"
python -m torch.distributed.launch --nproc_per_node=2 --master_port 2003 run.py --name ${name} --task ${task} --method ${method} ${base_setting} ${run_setting}

name="hyp_rn_v5.0_ss5_ep60_c2_topics"
weights="--step_prev_ckpt output/map_l6_increment_3tasks-ov/hyp_rn_v4.0_hiera_mh3_ep60_c2/step0/model_final.pth"
for t in 1 2; do
  incr_setting="--step ${t} --epoch 60 --lr 0.01 --window_stitching"
  python -m torch.distributed.launch --nproc_per_node=2 --master_port 2003 run.py --name ${name} --task ${task} --method ${method} ${incr_setting} ${weights} ${run_setting}
  weights=""
done

