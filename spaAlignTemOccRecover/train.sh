# # GaitSet
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=2 lib/main.py --cfgs ./config/gaitset.yaml --phase train

# # GaitPart
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 lib/main.py --cfgs ./config/gaitpart.yaml --phase train

# # GaitGL
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 lib/main.py --cfgs ./config/gaitgl.yaml --phase train

# # Baseline2d
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 lib/main.py --cfgs ./config/baseline2d.yaml --phase train

# # Baseline3d
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 lib/main.py --cfgs ./config/baseline3d.yaml --phase train

# # SpatialTemporalBackbone (STBackbone)
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 lib/main.py --cfgs ./config/stbackbone.yaml --phase train

# # STBackbone+SpaAlign
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 lib/main.py --cfgs ./config/spaAlign.yaml --phase train

# # SpaAlignTemOccRecover
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 lib/main.py --cfgs ./config/spaAlignTemOccRecover.yaml --phase train
