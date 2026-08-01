CUDA_VISIBLE_DEVICES=0 python -m main.main --dataset=vgg-fsd50k-dcase  --resume 4 \
    --domain_group=0 --data-root="/home/wakamatsu/DataSets2/FSD_VGG_DCASE" \
    --model=der --lr=1e-4 --resume=3 \
    --scheduler="cosine_annealing" \
    --n-epochs=100 --batch-size=128 --backbone=resnet18 \
    --loss=ce --epoch-scaling=const --alpha=0.5 --beta=0.5\
    --visualize --checkpoint --num-workers=8 \
    --buffer-size=400 --buffer-batch-size=64\
    --opt=adam --seed=1208 --wandb-name=Librispeech-DER \
    --is_audio=True
