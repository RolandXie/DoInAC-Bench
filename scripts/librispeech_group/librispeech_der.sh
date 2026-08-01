CUDA_VISIBLE_DEVICES=0 python -m main.main --dataset=librispeech \
    --domain_group=4 --model=der --lr=1e-4 \
    --scheduler="cosine_annealing" \
    --n-epochs=100 --batch-size=128 --backbone=resnet18 \
    --loss=ce --epoch-scaling=const --alpha=0.5 --beta=0.5\
    --visualize --checkpoint --num-workers=8 \
    --buffer-size=400 --buffer-batch-size=64\
    --opt=adam --seed=1208 --wandb-name=Librispeech-DER \
    --is_audio=True
