import os

import click
import numpy as np
import pickle as pkl
import yaml
from tqdm.auto import tqdm


X_MIN, X_MAX = 0, 51.2
Y_MIN, Y_MAX = -25.6, 25.6
Z_MIN, Z_MAX = -2, 4.4
VOXEL_SIZE = 0.2


@click.command()
@click.option(
    "--gen_label_root", 
    default=(
        "gpfsscratch/rech/kvd/uyl37fq/pasco_preprocess/gt_modified/instance_labels_v2"
    ),
)
@click.option("--output_dir", default="output/sequences")
@click.option("--config_path", default="semantic-kitti.yaml")
def main(gen_label_root: str, output_dir: str, config_path: str) -> None:

    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    learnning_map_inv = config["learning_map_inv"]
    label_map = np.empty(len(learnning_map_inv), dtype=np.uint32)
    label_map[list(learnning_map_inv)] = list(learnning_map_inv.values())

    sequence_dir_names = os.listdir(gen_label_root)
    sequence_prog_bar = tqdm(sequence_dir_names)
    for sequence_idx, sequence_dir_name in enumerate(
        sequence_prog_bar, start=1
    ):
        sequence_prog_bar.set_description(
            f"Sequence {sequence_idx}/{len(sequence_dir_names)}"
        )

        label_dir = os.path.join(gen_label_root, sequence_dir_name)
        label_names = os.listdir(label_dir)
        label_prog_bar = tqdm(label_names, leave=False)
        for label_idx, label_name in enumerate(label_prog_bar, start=1):
            label_prog_bar.set_description(
                f"Scan {label_idx}/{len(label_names)}"
            )

            label_path = os.path.join(label_dir, label_name)
            with open(label_path, "rb") as label_file:
                dense_labels = pkl.load(label_file)
            dense_semantic_labels = dense_labels["ssc_pred"]
            dense_instance_labels = dense_labels["pred_panoptic_seg"]

            discrete_coord_dir = os.path.join(
                output_dir, sequence_dir_name, "discrete_coords"
            )
            discrete_coord_path = (
                os.path.join(discrete_coord_dir, label_name)
                .replace("_1_1", "")
                .replace("pkl", "bin")
            )
            discrete_coords = (
                np.fromfile(discrete_coord_path, dtype=np.int32)
                .reshape(-1, 3)
            )
            x, y, z = discrete_coords.T
            sparse_semantic_labels = (
                dense_semantic_labels[0, x, y, z].astype(np.uint32)
            )
            sparse_semantic_labels = remap(sparse_semantic_labels, label_map)
            sparse_instance_labels = (
                dense_instance_labels[0, x, y, z].astype(np.uint32)
            )

            sparse_labels = (
                (sparse_instance_labels << 16)
                | (sparse_semantic_labels & 0xFFFF)
            )
            sparse_label_path = (
                discrete_coord_path
                .replace("discrete_coords", "predictions")
                .replace("bin", "label")
            )
            os.makedirs(os.path.dirname(sparse_label_path), exist_ok=True)
            sparse_labels.tofile(sparse_label_path)


def remap(labels: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    labels[labels == 255] = 0
    return label_map[labels]


if __name__ == "__main__":
    main()
