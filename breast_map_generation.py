# This tutorials generates 5 synthetic *T1-weighted* brain MRI scans from a label map.
# Specifically, it explains how to impose prior distributions on the GMM parameters, so that we can can generate images
# of desired intensity distribution.
# By default the GMM parameters (means and standard deviations of each Gaussian), are sampled from uniform distributions
# of wide predefined ranges, thus yielding output images of random contrast.
# Here we show how to generate images of desired contrast by imposing the prior distributions from which we sample the
# means and standard deviations of the GMM.

import os
import time
from lab2im import utils
from lab2im.image_generator import ImageGenerator

import numpy as np
def printnpy(filename):
    file = np.load(filename)
    print(filename + ': ' + str(file))
    return None

# labels: [background=0, breast=1, chest=2, skin=3, fgt=4, fat=5]

# label map to generate images from
path_label_map = 'base_images_phantom4/breast_label_map.nii.gz'

# general parameters
n_examples = 5
result_dir = './generated_images'
output_shape = None  # shape of the output images, obtained by randomly cropping the generated images

# specify structures from which we want to generate
generation_list = np.array([0, 1, 2, 3, 4, 5])
np.save('./base_images_phantom4/generation_labels.npy', generation_list)
generation_labels = './base_images_phantom4/generation_labels.npy'
# we regroup structures into K classes, so that they share the same distribution for image generation

# We specify here that we type of prior distributions to sample the GMM parameters.
# By default prior_distribution is set to 'uniform', and in this example we want to change it to 'normal'.
prior_distribution = 'uniform'
# We specify here the hyperparameters of the prior distributions to sample the means of the GMM.
# As these prior distributions are Gaussians, they are each controlled by a mean and a standard deviation.
# Therefore, the numpy array pointed by prior_means is of size (2, K), where K is the nummber of classes specified in
# generation_classes. The first row of prior_means correspond to the means of the Gaussian priors, and the second row
# correspond to standard deviations.

# labels: [background=0, breast=1, chest=2, skin=3, fgt=4, fat=5]
means_list = np.array([np.array([0., 0.8, 0.8,  0.0, 0.8, 0.0]), # Rho_w values
                       np.array([0., 1.0, 1.0,  0.2, 1.0, 0.1]),
                       np.array([0., 0.0, 0.0,  0.8, 0.0, 0.9]), # Rho_f values
                       np.array([0., 0.2, 0.2,  1.0, 0.2, 1.0]),
                       np.array([10000., 1000., 1000., 196.,  1166., 196.]) * (10**-3), # T1 Values
                       np.array([10001., 1200., 1200., 396.,  1366., 396.]) * (10**-3),
                       np.array([10000., 1000., 1000., 48.,    250.,  48.]) * (10**-3), # T2 Values
                       np.array([10001., 1200., 1200., 68.,    450.,  68.]) * (10**-3),
                       ])
np.save('./base_images_phantom4/prior_means.npy', means_list)
prior_means = './base_images_phantom4/prior_means.npy'
# same as for prior_means, but for the standard deviations of the GMM. mean and std of the intra-phantom values
stds_list =  np.array([np.array([0, 0, 0, 0, 0, 0]).astype(float), # Rho_w values
                       np.array([0, 0, 0, 0, 0, 0]).astype(float),
                       np.array([0, 0, 0, 0, 0, 0]).astype(float), # Rho_f values
                       np.array([0, 0, 0, 0, 0, 0]).astype(float),
                       np.array([0, 0, 0, 0, 0, 0]).astype(float) * (10**-3), # T1 Values
                       np.array([0, 0, 0, 0, 0, 0]).astype(float) * (10**-3),
                       np.array([0, 0, 0, 0, 0, 0]).astype(float) * (10**-3), # T2 Values
                       np.array([0, 0, 0, 0, 0, 0]).astype(float) * (10**-3),
                       ])
np.save('./base_images_phantom4/prior_stds.npy', stds_list)
prior_stds = './base_images_phantom4/prior_stds.npy'

########################################################################################################
# instantiate BrainGenerator object
brain_generator = ImageGenerator(labels_dir=path_label_map,
                                 generation_labels=generation_labels,
                                 prior_distributions=prior_distribution,
                                 prior_means=prior_means,
                                 n_channels=4,
                                 prior_stds=prior_stds,
                                 output_shape=output_shape,
                                 use_specific_stats_for_channel=True)

# create result dir
utils.mkdir(result_dir)

for n in range(n_examples):

    # generate new image and corresponding labels
    start = time.time()
    im, lab = brain_generator.generate_image()
    end = time.time()
    print('generation {0:d} took {1:.01f}s'.format(n, end - start))

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_%s.nii.gz' % n))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 't1_labels_%s.nii.gz' % n))

