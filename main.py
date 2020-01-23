#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import numpy as np

import modules

def main():
	# retrieve path to csv files as input
	path = str(input()).rstrip()

	data_tools = modules.DataTools()

if __name__ == "__main__":
	main()



# arrs_test_1 = [to_array(grayscale_gleam(img)) 
#                for img in imgs_test_1]


# energies_test_1 = np.stack([laplace(arr, 3, square=True) for arr in arrs_test_1], axis=2)
# energies_test_1.shape


# # This allows us to get the index of the best-matching energy image per pixel like so:
# print(energies_test_1.shape)

# highest_energies_idx = np.argmax(energies_test_1, axis=2)
# highest_energies_idx

# # We can now use this information to pick the correct pixels. First we need the images we're sampling from as an array, so that we can can index the pixels easily:
# source = np.stack(arrs_test_1, axis=0)
# source.shape

# # We can now synthesize an image by sampling from each layer according to it's highest energy.
# fused_array = np.zeros(energies_test_1.shape[0:2], dtype=np.float32)
# rows, cols = fused_array.shape
# for row in range(rows):
#     for col in range(cols):
#         idx = highest_energies_idx[row, col]
#         fused_array[row, col] = source[idx, row, col]


# # This is how it looks:

# to_image(fused_array)


# # We can do the same with a color image:
# source_color = np.stack(imgs_test_1, axis=0)
# source_color.shape

# height, width = energies_test_1.shape[0:2]
# fused_array_color = np.zeros((height, width, 3), dtype=np.float32)
# rows, cols = fused_array.shape
# for row in range(rows):
#     for col in range(cols):
#         idx = highest_energies_idx[row, col]
#         fused_array_color[row, col, :] = source_color[idx, row, col, :]

# to_image(fused_array_color)

# # Although the whole image now appears to be in focus, the color image makes sampling artifacts much more visible. Multiple problems are at play:
# # 
# # - Due to movement of the camera, each image might show a slightly area of the object
# # - The image magnification might be different between different focal lengths
# # - Noise in the image introduces gradients that do not belong to the object.

# # ## Depth from focus
# # 
# # Rather than synthesizing an in-focus image, we can also do the opposite and synthesize a depth image. Indeed we're already halfway there: We know that the images are already sorted according to their focal length. If they weren't, we would have to take additional information into account, such as EXIF information stored in the image.
# # 
# # Because the images are sorted, the layer index we picked using `argmax` already encodes a depth, albeit not a (physically) meaningful one. Let's visualize that.

# normalized_indexes = normalize(highest_energies_idx)

# # We slightly blur the image to not get too distracted by noise.
# normalized_indexes_blurred = skimage.filters.gaussian(normalized_indexes, 1.4)
# to_image(normalized_indexes_blurred)


# # As it turns out, this representation is far from perfect:
# # 
# # - Items that are clearly in the background (i.e. the top of the example image) are white and thus appear to be in the foreground, while
# # - items that are clearly in the foreground (e.g. the pearl) are dark and thus appear to be in the background.
# # 
# # The reason here is that areas without sufficient texture don't have any strong energy regardless of their distance. While this doesn't have a big effect on the pixel color, it does effect the depth synthesis. Likewise, noise creates an appearance of energy where there is none in the actual object, resulting in background items to be sampled.