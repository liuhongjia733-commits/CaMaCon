# CaMaCon

This package contains the CaMaCon implementation and the pretrained weights for two invertible autoencoders and five point-cloud classifiers on each dataset.

## Install dependencies

Use a Python environment with PyTorch and a matching PyTorch3D build. A CUDA-capable GPU is recommended. Follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and the [PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), then install the remaining dependencies.

```bash
python -m pip install numpy h5py loguru matplotlib tqdm
python -c "import torch; from pytorch3d.ops import knn_points; print(torch.__version__, torch.cuda.is_available())"
```

## Download and prepare the datasets

| Dataset | Download | Destination |
| --- | --- | --- |
| ModelNet40 | [Stanford HDF5 archive](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) | `data/modelnet40_ply_hdf5_2048/` |
| ShapeNetPart | [Stanford HDF5 archive](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip) | `data/shapenetpart_hdf5_2048/` |

Both destinations are relative to this README. Extract the archives and put their HDF5 shards and file lists directly in the corresponding destination. The ShapeNetPart archive uses an `hdf5_data` folder. Rename it to `shapenetpart_hdf5_2048` when moving it under `data/`.

In each dataset directory, make sure `train_files.txt` and `test_files.txt` each list one shard filename per line, in the original order. Remove any leading directory components from the list entries. For example, an entry should read `ply_data_test0.h5`. If a file list is missing, create it from the matching numbered shards in ascending order. Leave the pretrained weights in their packaged locations.

## Run CaMaCon

From the directory containing this README, run a short smoke test.

```bash
python protocols/attackers/run_camacon.py --dataset ModelNet40 --model pointnet --batch_size 2 --num_batches_to_attack 1 --num_iter 2 --binary_step 1 --tag smoke
```

Run the full evaluation with the built-in experiment parameters.

```bash
python protocols/attackers/run_camacon.py --dataset ModelNet40 --model pointnet --batch_size 32 --num_batches_to_attack 17 --centroid_align --tag full10
```

Choose either `ModelNet40` or `ShapeNetPart` and any of `pointnet`, `pointnet2`, `dgcnn`, `pct`, and `curvenet`. Results and a JSON metric summary are written under `attacked_data/camacon_centroid/` by default. Use `--help` for available options, including `--output_dir`.
