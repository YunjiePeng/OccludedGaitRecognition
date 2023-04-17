1. Prepare the silhouettes for generating static occlusions.
① Download mapillary-vistas-dataset_public_v2.0.zip from https://www.mapillary.com/dataset/vistas
② Run static_occlusion_filter_for_mapillary-vistas.py to transform static occlusions (Bench, Bicycle, FireHydrant, Motorcycle, Pole, and TrashCan) into silhouettes.
③ Use the image names given by the Excel file to select silhouettes and organize them into required directories for generating static occlusions.

2. Run generate_occlusion_withFrameOccLabel.py to generate the OccCASIA-B dataset.
