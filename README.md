# Bright-Field Microscopy Image Fusion Prototype

[![Continuous Integration](https://github.com/vaugus/bright-field-microscopy-image-fusion-prototype/actions/workflows/ci.yaml/badge.svg)](https://github.com/vaugus/bright-field-microscopy-image-fusion-prototype/actions/workflows/ci.yaml)

A Python implementation of an image fusion method using the Laplacian of Gaussian Energy approach.

## Fusion Rule

The first step of the fusion rule is to extract the edges of the images with the Laplacian filter. Then, the indices for the highest energies of laplacian are calculated for each pixel and associated to one of the images in the stack. The final image is composed by choosing pixels from the layers with highest energy values.

## Evaluation of fusion results

We used the following evaluation methods, as proposed by ([NAIDU; RAOL, 2008](#references)): the spatial frequency index, the standard deviation of the histogram of the image, and the image entropy.

## Usage

```sh
python3 main.py < /path/to/image/dataset
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## References

Naidu, V. P. S., and Jitendra R. Raol. "Pixel-level image fusion using wavelets and principal component analysis." Defence Science Journal 58.3 (2008): 338.
