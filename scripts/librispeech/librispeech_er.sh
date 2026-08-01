CUDA_VISIBLE_DEVICES=0 python -m main.main --dataset=librispeech \
    --domain_group=2 --model=er --lr=1e-4 \
    --scheduler="cosine_annealing" \
    --n-epochs=100 --batch-size=128 --backbone=resnet18 \
    --loss=ce --epoch-scaling=const \
    --buffer-size=400 --buffer-batch-size=64 \
    --visualize --checkpoint --num-workers=8 \
    --opt=adam --seed=1208 --wandb-name=Librispeech-ER \
    --is_audio=True
