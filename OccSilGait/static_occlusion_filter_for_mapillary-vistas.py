from __future__ import print_function
import os
import os.path as osp
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt, patches

from PIL import Image
import cv2

parser = argparse.ArgumentParser(description="Mapillary Vistas demo")
parser.add_argument("--v1_2", action="store_true", default=True,
                    help="If true, the demo version of the dataset is 1.2, else 2.0")

image_list = sorted(os.listdir("./training/images/"))
static_occ_categories = [
    "object--bench", 
    "object--fire-hydrant",
    # "object--junction-box",
    "object--mailbox",
    # "object--support--pole",
    # "object--support--utility-pole",
    # "object--trash-can",
    "object--vehicle--bicycle",
    "object--vehicle--motorcycle",
]

static_occ_dict = {
    "object--bench": {"readable": "Bench", "color": [250, 0, 30]},
    "object--fire-hydrant": {"readable": "Fire Hydrant", "color": [100, 170, 30]},
    "object--junction-box": {"readable": "Junction Box", "color": [40, 40, 40]}, 
    "object--mailbox": {"readable": "Mailbox", "color": [33, 33, 33]}, 
    "object--support--pole": {"readable": "Pole", "color": [153, 153, 153]}, 
    "object--support--utility-pole": {"readable": "Utility Pole", "color": [0, 0, 80]},
    "object--trash-can": {"readable": "Trash Can", "color": [140, 140, 20]},  
    "object--vehicle--bicycle": {"readable": "Bicycle", "color": [119, 11, 32]}, 
    "object--vehicle--motorcycle": {"readable": "Motorcycle", "color": [0, 0, 230]}, 
}

def load_labels(version):
    # read in config file
    with open('config_{}.json'.format(version)) as config_file:
        config = json.load(config_file)
    # in this example we are only interested in the labels
    labels = config['labels']

    # print labels
    # print("There are {} labels in the config file".format(len(labels)))
    # for label_id, label in enumerate(labels):
    #     print("{:>30} ({:2d}): {:<40} has instances: {}".format(label["readable"], label_id, label["name"], label["instances"]))

    return labels

def read_panoptic_json(args):
    # read in panoptic file
    if args.v1_2:
        panoptic_json_path = "training/v1.2/panoptic/panoptic_2018.json"
    else:
        panoptic_json_path = "training/v2.0/panoptic/panoptic_2020.json"
    with open(panoptic_json_path) as panoptic_file:
        panoptic = json.load(panoptic_file)

    # convert annotation infos to image_id indexed dictionary
    panoptic_per_image_id = {}
    for annotation in panoptic["annotations"]:
        panoptic_per_image_id[annotation["image_id"]] = annotation

    # convert category infos to category_id indexed dictionary
    panoptic_category_per_id = {}
    for category in panoptic["categories"]:
        panoptic_category_per_id[category["id"]] = category
    
    return panoptic_per_image_id, panoptic_category_per_id

def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array

def generate_target_mask(image, target_color):
    image_r = image[:, :, 0].copy().astype(np.uint32)
    image_g = image[:, :, 1].copy().astype(np.uint32)
    image_b = image[:, :, 2].copy().astype(np.uint32)

    image_coding = image_r * 256 * 256 + image_g * 256 + image_b
    target_color_coding = target_color[0] * 256 * 256 + target_color[1] * 256 + target_color[2]
    target_mask = (image_coding == target_color_coding)
    return target_mask

def main(args, labels, panoptic_per_image_id, panoptic_category_per_id, image_id):
    # set up paths for rgb image and panoptic segmentation image.
    image_path = "training/images/{}.jpg".format(image_id)
    label_path = "training/{}/labels/{}.png".format(version, image_id)
    panoptic_path = "training/{}/panoptic/{}.png".format(version, image_id)

    # load images
    base_image = Image.open(image_path)
    label_image = Image.open(label_path)
    panoptic_image = Image.open(panoptic_path)


    # plot the result
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,15))

    ax[0].imshow(base_image)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title("Base image")

    ax[1].imshow(base_image)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title("Detect image")
    
    ax[2].imshow(label_image.convert("RGB"))
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_title("Labels")
    ax[2].axis('off')

    fig.tight_layout()
    # fig, ax = plt.subplots()
    # ax.imshow(base_image)

    if osp.exists(des_path) is False:
        os.makedirs(des_path)

    # Panoptic json file handling
    # convert segment infos to segment id indexed dictionary
    cur_panoptic = panoptic_per_image_id[image_id]
    cur_segments = {}
    for segment_info in cur_panoptic["segments_info"]:
        cur_segments[segment_info["id"]] = segment_info

    print('Panoptic segments:')
    panoptic_array = np.array(panoptic_image).astype(np.uint32)
    panoptic_id_array = panoptic_array[:,:,0] + (2**8)*panoptic_array[:,:,1] + (2**16)*panoptic_array[:,:,2]
    panoptic_ids_from_image = np.unique(panoptic_id_array)
    for panoptic_id in panoptic_ids_from_image:
        if panoptic_id == 0:
            # void image areas don't have segments
            continue

        segment_info = cur_segments[panoptic_id]
        category = panoptic_category_per_id[segment_info["category_id"]]
        if category['supercategory'] not in static_occ_categories:
            continue

        print("image-{}-segment {:8d}: label {:<40}, area {:6d}, bbox {}".format(
            image_id,
            panoptic_id,
            category["supercategory"],
            segment_info["area"],
            segment_info["bbox"],
        ))

        x, y, w, h = segment_info["bbox"]
        segment_rgb = base_image.crop((x, y, x+w, y+h))
        segment_label = label_image.crop((x, y, x+w, y+h))
        
        segment_sil = np.array(segment_label.convert("RGB"))
        target_color = static_occ_dict[category['supercategory']]['color']
        target_mask = generate_target_mask(segment_sil, target_color)

        segment_sil[target_mask] = 255
        segment_sil[target_mask == False] = 0
        segment_sil = Image.fromarray(segment_sil)

        save_dir = static_occ_dict[category['supercategory']]['des_dir']
        if osp.exists(save_dir) is False:
            os.makedirs(save_dir)

        save_rgb_name = "{}-{}-{}-rgb.png".format(image_id, category['supercategory'], panoptic_id)
        segment_rgb.save(osp.join(save_dir, save_rgb_name))
        save_label_name = "{}-{}-{}-label.png".format(image_id, category['supercategory'], panoptic_id)
        segment_label.save(osp.join(save_dir, save_label_name))
        save_sil_name = "{}-{}-{}-sil.png".format(image_id, category['supercategory'], panoptic_id)
        segment_sil.save(osp.join(save_dir, save_sil_name))

        target_color = np.array(target_color) / 255
        ax[1].text(x=x, y=y, s=str(panoptic_id), 
                color='white', verticalalignment="top", 
                bbox={"color": target_color,"pad": 0})
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=target_color, facecolor='none')
        ax[1].add_patch(rect)
                
        # import pdb
        # pdb.set_trace()
        # remove to show every segment is associated with an id and vice versa
        cur_segments.pop(panoptic_id)

    if osp.exists(osp.join(des_path, 'labels')) is False:
        os.makedirs(osp.join(des_path, 'labels'))
    fig.savefig(osp.join(des_path, 'labels', '{}.png'.format(image_id)))
    plt.close()
    # every positive id is associated with a segment
    # after removing every id occuring in the image, the segment dictionary is empty
    # assert len(cur_segments) == 0
        

image_list = image_list[:1000]
des_path = "./results0kto1k/"

# image_list = image_list[1000:2000]
# des_path = "./results1kto2k/"

# image_list = image_list[2000:3000]
# des_path = "./results2kto3k/"

# image_list = image_list[3000:4000]
# des_path = "./results3kto4k/"

# image_list = image_list[4000:5000]
# des_path = "./results4kto5k/"

# image_list = image_list[5000:6000]
# des_path = "./results5kto6k/"

# image_list = image_list[6000:7000]
# des_path = "./results6kto7k/"

# image_list = image_list[7000:8000]
# des_path = "./results7kto8k/"

if __name__ == "__main__":
    args = parser.parse_args()

    version = "v1.2" if args.v1_2 else "v2.0"
    labels = load_labels(version)
    panoptic_per_image_id, panoptic_category_per_id = read_panoptic_json(args)

    for key in static_occ_dict.keys():
        value = static_occ_dict[key]
        value['des_dir'] = osp.join(des_path, value['readable'])
        static_occ_dict[key] = value
    print("static_occ_dict:", static_occ_dict)
    print("static_occ_categories:", static_occ_categories)

    img_count = 1
    for img in image_list:
        img_id = img.split('.')[0]
        print("\nimg_count:{}, img_id:{}".format(img_count, img_id))
        main(args, labels, panoptic_per_image_id, panoptic_category_per_id, img_id)
        img_count += 1
