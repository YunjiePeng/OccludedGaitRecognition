1. Prepare the silhouettes for generating static occlusions.
① Download mapillary-vistas-dataset_public_v2.0.zip from https://www.mapillary.com/dataset/vistas
② Run static_occlusion_filter_for_mapillary-vistas.py to transform static occlusions (Bench, Bicycle, FireHydrant, Motorcycle, Pole, and TrashCan) into silhouettes.
③ Use the image names given by the csv file to select silhouettes and organize them as follows for generating static occlusions:

static_occlusions---Bench---"10LozR-r_PnVyUloTy7P_A-object--bench-3490375-sil.png"
                         ---"18UTnNtyNaTlJ7kVcqs1vw-object--bench-2501428-sil.png"
                         ...
                         
                  ---Bicycle---"-3Ce66lOpbY2j4Tgngr-dQ-object--vehicle--bicycle-6579813-sil.png"
                            ---"-6kI3IofoH9vQ7kM6ake2w-object--vehicle--bicycle-5592154-sil.png"
                            ...
                  ...
                  

2. Run generate_occlusion_withFrameOccLabel.py to generate the OccCASIA-B dataset.
