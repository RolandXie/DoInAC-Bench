CUDA_VISIBLE_DEVICES=0 python -m main.main --dataset=librispeech --resume 5\
    --domain_group=2 --model=finetune --lr=1e-4 \
    --scheduler="cosine_annealing" \
    --n-epochs=1 --batch-size=128 --backbone=resnet18 \
    --loss=ce --epoch-scaling=const \
    --visualize --checkpoint --num-workers=8 \
    --opt=adam --seed=1208 \
    --wandb-name=Librispeech-FINETUNE \
    --is_audio=True
