CUDA_VISIBLE_DEVICES=0 python -m main.main --dataset=vgg-fsd50k-dcase \
    --domain_group=0 --data-root="/home/wakamatsu/DataSets2/FSD_VGG_DCASE" \
    --model=lwf --lr=1e-4 --resume=3 \
    --scheduler="cosine_annealing" \
    --n-epochs=100 --batch-size=128 --backbone=resnet18 \
    --loss=ce --epoch-scaling=const --alpha=1 --scale='linear'\
    --visualize --checkpoint --num-workers=8 \
    --opt=adam --seed=1208 --wandb-name=Librispeech-LWF \
    --is_audio=True
