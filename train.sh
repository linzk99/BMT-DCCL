# ACDC  
CUDA_VISIBLE_DEVICES=0 python -u ./code/train_BMT_DCCL.py --exp ACDC/BMT_DCCL --labeled_num 7 --cutmix_prob 0.75 --root_path ../data/ACDC --num_classes 4

# PROMISE
CUDA_VISIBLE_DEVICES=0 python -u ./code/train_BMT_DCCL.py --exp Prostate/BMT_DCCL --labeled_num 7 --cutmix_prob 1 --root_path ../data/Prostate --num_classes 2


