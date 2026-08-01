CUDA_VISIBLE_DEVICES=0 python -m main.main --dataset=librispeech \
    --domain_group=4 --model=lwf --lr=1e-4 \
    --scheduler="cosine_annealing" \
    --n-epochs=100 --batch-size=128 --backbone=resnet18 \
    --loss=ce --epoch-scaling=const --alpha=1 --scale='linear'\
    --visualize --checkpoint --num-workers=8 \
    --opt=adam --seed=1208 --wandb-name=Librispeech-LWF \
    --is_audio=True
