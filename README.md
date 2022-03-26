# Deep Multi Depth Panoramas for View Synthesis

Official PyTorch Implementation of paper "Deep Multi Depth Panoramas for View Synthesis", ECCV 2020.


[Kai-En Lin](https://cseweb.ucsd.edu/~k2lin/)<sup>1</sup> 	[Zexiang Xu](https://cseweb.ucsd.edu/~zex014/)<sup>1,3</sup> 	[Ben Mildenhall](https://bmild.github.io/)<sup>2</sup> 	[Pratul P. Srinivasan](https://pratulsrinivasan.github.io/)<sup>2</sup>
[Yannick Hold-Geoffroy](https://yannickhold.com/)<sup>3</sup> 	[Stephen DiVerdi](https://www.stephendiverdi.com/)<sup>3</sup> 	[Qi Sun](https://qisun.me/)<sup>3</sup> 	[Kalyan Sunkavalli](http://www.kalyans.org/)<sup>3</sup> 	[Ravi Ramamoorthi](https://cseweb.ucsd.edu/~ravir/)<sup>1</sup>
<sup>1</sup>University of California, San Diego 	<sup>2</sup>University of California, Berkeley 	<sup>3</sup>Adobe Research

## Requirements

* PyTorch & torchvision

* numpy

* imageio

* matplotlib

## Usage

We only provide the inference code.

For training code, please refer to this repo, [Deep 3D Mask Volume for View Synthesis of Dynamic Scenes](https://github.com/ken2576/deep-3dmask), in `train_mpi` directory.

1. run ```python gen_mpi.py --scene cafe/ --out example_cafe/ --model_path ckpts/paper_model.pth ```

2. run ```python gen_ldp.py --scene cafe/ --mpi_folder example_cafe/ --ldp_folder example_cafe_ldp/ --out_folder example_cafe_img```

Note:
You might need to implement custom camera poses for rendering. Some functions are in ```gen_ldp.py```.

The extrinsics are in world to camera convention.

For custom data, you could pack the data similar to ```cafe/```.

The camera poses are in the same format as [Local Light Field Fusion](https://github.com/Fyusion/LLFF), meaning that it is in (N, 17), N is the number of source views.

The 17-dim vector is composed of 3x5 matrix (just do ```np.reshape(3, 5)```) and 2-dim vector for near and far plane bounds.
The 3x5 matrix is 3x4 `[R|t]` from camera extrinsics and last column is `(height, width, focal length)`.


## Citation

```
@inproceedings{lin2020mdp,
  title={Deep Multi Depth Panoramas for View Synthesis},
  author={Lin, Kai-En and Xu, Zexiang and Mildenhall, Ben and Srinivasan, Pratul P and Hold-Geoffroy, Yannick and DiVerdi, Stephen and Sun, Qi and Sunkavalli, Kalyan and Ramamoorthi, Ravi},
  year={2020},
  booktitle={ECCV},
}
```