import os

import click
import MinkowskiEngine as ME
import numpy as np
import yaml
from tqdm.auto import tqdm


X_MIN, X_MAX = 0, 51.2
Y_MIN, Y_MAX = -25.6, 25.6
Z_MIN, Z_MAX = -2, 4.4
VOXEL_SIZE = 0.2


@click.command()
@click.option("--root_dir", default="gpfsdswork/dataset/SemanticKITTI/dataset/sequences")
@click.option("--config_path", default="semantic-kitti.yaml")
@click.option("--output_dir", default="output/sequences")
def main(root_dir: str, config_path: str, output_dir: str) -> None:

    # Load config
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    sequences = config["split"]["train"] + config["split"]["valid"]

    sequence_prog_bar = tqdm(sequences)
    for sequence_idx, sequence in enumerate(sequence_prog_bar, start=1):
        sequence_prog_bar.set_description(f"Sequence {sequence_idx}/{len(sequences)}")

        scan_dir = os.path.join(root_dir, f"{sequence:02}", "velodyne")
        scan_names = os.listdir(scan_dir)
        scan_prog_bar = tqdm(scan_names, leave=False)
        for scan_idx, scan_name in enumerate(scan_prog_bar, start=1):
            scan_prog_bar.set_description(f"Scan {scan_idx}/{len(scan_names)}")

            # Load scan
            scan_path = os.path.join(scan_dir, scan_name)
            scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            coords = scan[:, :3]
            feats = scan[:, 3:]

            # Load labels
            label_path = (
                scan_path
                .replace("velodyne", "labels")
                .replace("bin", "label")
            )
            labels = np.fromfile(label_path, dtype=np.uint32)
            semantic_labels = labels & 0xFFFF
            instance_labels = labels >> 16

            # Quantize
            x, y, z = coords.T
            keep = (
                (x >= X_MIN) & (x < X_MAX) &
                (y >= Y_MIN) & (y < Y_MAX) &
                (z >= Z_MIN) & (z < Z_MAX)
            )
            min_bound = np.array([X_MIN, Y_MIN, Z_MIN])
            discrete_coords, mapping = ME.utils.sparse_quantize(
                coords[keep] - min_bound,
                return_index=True,
                quantization_size=VOXEL_SIZE,
            )
            unique_coords = coords[keep][mapping]
            unique_feats = feats[keep][mapping]
            unique_semantic_labels = semantic_labels[keep][mapping]
            unique_instance_labels = instance_labels[keep][mapping]

            # Save discrete coords
            discrete_coords_path = (
                scan_path
                .replace(root_dir, output_dir)
                .replace("velodyne", "discrete_coords")
            )
            os.makedirs(os.path.dirname(discrete_coords_path), exist_ok=True)
            discrete_coords.numpy().astype(np.int32).tofile(discrete_coords_path)

            # Save scan
            output_scan = np.hstack([unique_coords, unique_feats])
            output_scan_path = scan_path.replace(root_dir, output_dir)
            os.makedirs(os.path.dirname(output_scan_path), exist_ok=True)
            output_scan.tofile(output_scan_path)

            # Save labels
            output_labels = (
                (unique_instance_labels << 16)
                | (unique_semantic_labels & 0xFFFF)
            )
            output_label_path = label_path.replace(root_dir, output_dir)
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            output_labels.tofile(output_label_path)


if __name__ == "__main__":
    main()
