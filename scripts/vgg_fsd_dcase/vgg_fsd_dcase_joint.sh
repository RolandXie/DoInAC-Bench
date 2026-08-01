CUDA_VISIBLE_DEVICES=0 python -m main.main --dataset=vgg-fsd50k-dcase \
    --data-root="/home/wakamatsu/DataSets2/FSD_VGG_DCASE" --domain_group=0 \
    --model=joint --lr=1e-4 \
    --scheduler="cosine_annealing"  \
    --n-epochs=1 --batch-size=128 --backbone=resnet18 \
    --loss=ce --epoch-scaling=const \
    --visualize --checkpoint --num-workers=8 \
    --opt=adam --seed=1208 \
    --wandb-name=Librispeech-JOINT \
    --is_audio=True
