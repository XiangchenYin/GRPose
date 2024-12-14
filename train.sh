cd /mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/xcyin/GRPose

start=$(date +%s)

# export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

model="grpose"
dataset="humanart"
device="-7-L40S"

# MASTER_ADDR=$CHIEF_IP MASTER_PORT=10435 WORLD_SIZE=$HOST_NUM NODE_RANK=$INDEX \

/mnt/pfs-mc0p4k/tts/team/digital_avatar_group/sunwenzhang/xcyin/xcyin_dit/bin/python \
 train.py --config configs/${model}/${dataset}.yaml --max_epochs 40 \
--control_ckpt models/init_grpose.ckpt --devices 7 --scale_lr false \
2>&1  | tee "logs/${model}_${dataset}${device}.log"

end=$(date +%s)

runtime=$((end-start))

echo "Total Execution Time $((runtime/3600)) hours!!!"
