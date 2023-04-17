# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=1123 --nproc_per_node=8 lib/main.py --cfgs ./config/gaitpart.yaml --phase test

# # GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=1123 --nproc_per_node=8 lib/main.py --cfgs ./config/gaitgl.yaml --phase test

# # Baseline2d
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=1123 --nproc_per_node=8 lib/main.py --cfgs ./config/baseline2d.yaml --phase test

# # Baseline3d
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=1123 --nproc_per_node=8 lib/main.py --cfgs ./config/baseline3d.yaml --phase test

# # SpatialTemporalBackbone (STBackbone)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1123 lib/main.py --cfgs ./config/stbackbone.yaml --iter 40000 --phase test

# # STBackbone+SpaAlign
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1123 lib/main.py --cfgs ./config/spaAlign.yaml --iter 40000 --phase test

# # SpaAlignTemOccRecover
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1123 lib/main.py --cfgs ./config/spaAlignTemOccRecover.yaml --iter 40000 --phase test
