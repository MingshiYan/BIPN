#!/bin/bash


# lr=('0.001' '0.005' '0.01')
# lr=('0.001' '0.0001' '0.0005')
lr=('0.001')
# lr=('0.01')
# reg_weight=('0.01' '0.001' '0.0001')
emb_size=(64)
# lr=('0.0005')
reg_weight=('0.001')
# log_regs=('0.5' '0.7' '1.0')
# log_regs=('0.9' '0.7' '0.5')
# log_regs=('1.0' '1.2' '1.5')
log_regs=('0.3')

dataset=('ml')
device='cuda:4'
batch_size=1024
decay=('0')
model_name='model'
gpu_no=1
neg_count=4


for name in ${dataset[@]}
do
    for l in ${lr[@]}
    do
        for reg in ${reg_weight[@]}
        do
            for emb in ${emb_size[@]}
            do
            for dec in ${decay[@]}
            do
            for log_reg in ${log_regs[@]}
            do
                echo 'start train: '$name
                `
                    python main.py \
                        --lr ${l} \
                        --reg_weight ${reg} \
                        --log_reg ${log_reg} \
                        --data_name $name \
                        --embedding_size $emb \
                        --device $device \
                        --decay $dec \
                        --batch_size $batch_size \
                        --gpu_no $gpu_no \
                        --neg_count $neg_count \
                        --model_name $model_name
                `
                echo 'train end: '$name
            done
            done
            done
        done
    done
done