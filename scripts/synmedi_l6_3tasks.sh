gpu_setting="--MASTER_PORT 21577 --MASTER_ADDRESS 10.4.28.207"
task="configs/synmedi/task_files/l6_increment_3tasks"
method="TOPICS+"

name="hyp_rn_v5.0_ss5_30_c3_dice0.7_topics+"
run_setting="--dataset synmedi --batch_size 24"
base_setting="--step 0 --epoch 30 --lr 0.03"
python -m torch.distributed.launch --nproc_per_node=3 --master_port 3115 run.py ${gpu_setting} --name ${name} --task ${task} --method ${method} ${base_setting} ${run_setting}

for t in 1 2; do
  incr_setting="--step ${t} --epoch 30 --lr 0.02 --distance_sim_weight 10 --triplet_loss_weight 0.01"
  python -m torch.distributed.launch --nproc_per_node=3 --master_port 3115 run.py ${gpu_setting} --name ${name} --task ${task} --method ${method} ${incr_setting} ${run_setting}
  weights=""
done

method="TOPICS"
name="hyp_rn_v5.0_ss5_30_c3_dice0.7_topics"
run_setting="--dataset synmedi --batch_size 24"
base_setting="--step 0 --epoch 30 --lr 0.03"
python -m torch.distributed.launch --nproc_per_node=3 --master_port 3115 run.py ${gpu_setting} --name ${name} --task ${task} --method ${method} ${base_setting} ${run_setting}

for t in 1 2; do
  incr_setting="--step ${t} --epoch 30 --lr 0.02 --distance_sim_weight 10 --triplet_loss_weight 0.01"
  python -m torch.distributed.launch --nproc_per_node=3 --master_port 3115 run.py ${gpu_setting} --name ${name} --task ${task} --method ${method} ${incr_setting} ${run_setting}
  weights=""
done