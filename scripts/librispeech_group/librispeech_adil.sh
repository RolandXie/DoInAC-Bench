CUDA_VISIBLE_DEVICES=0  python -m main.main --dataset=librispeech \
    --domain_group=4 --n_task=6 --model=adil --lr=1e-4 \
    --scheduler="cosine_annealing"  \
    --n-epochs=100 --batch-size=128 --backbone=resnet18_ds_bn_cl \
    --loss=ce --epoch-scaling=const \
    --visualize --checkpoint --num-workers=8 \
    --opt=adam --seed=1208 \
    --wandb-name=Librispeech-FINETUNE \
    --is_audio=True
