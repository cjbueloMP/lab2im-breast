# This script shows a usecase of the ImageGenerator wrapper around the lab2im_model.
# It generates 5 synthetic T1-weighted brain MRI scans from a label map.
# Images are generated by sampling a GMM conditioned on this label map. The parameters of the GMM (means and variances)
# are randomly sampled for each image from normal prior distributions specified in prior_means and prior_stds.
# Here the prior distributions are heavily peaked around T1w intenisty distributions.
# We also perform some data augmentation (spatial transformation, random cropping, random blurring, bias field, gamma
# augmentation), so that the generated images are all different.

# python imports
import os

# project imports
from lab2im import utils
from lab2im.image_generator import ImageGenerator

# label map to generate images from
path = '../data_example/brain_label_map.nii.gz'

# general parameters
n_examples = 5
output_shape = None  # shape of the output images, obtained by randomly cropping the sampled image
result_folder = '../data_example/generated_images'

# specify structures from which we want to generate
generation_labels = '../data_example/generation_labels.npy'
# specify structures that we want to keep in the output label maps
segmentation_labels = '../data_example/segmentation_labels.npy'
# we regroup structures into classes, so that they are generated from the same distribution
generation_classes = '../data_example/generation_classes.npy'

# type of prior distribution to use when sampling the gmm parameters
prior_distribution = 'normal'
# hyperparameters controlling the prior distributions of the gmm
prior_means = '../data_example/prior_means.npy'
prior_stds = '../data_example/prior_stds.npy'

########################################################################################################

# instantiate BrainGenerator object
brain_generator = ImageGenerator(labels_dir=path,
                                 generation_labels=generation_labels,
                                 output_labels=segmentation_labels,
                                 generation_classes=generation_classes,
                                 prior_means=prior_means,
                                 prior_stds=prior_stds,
                                 prior_distributions=prior_distribution,
                                 output_shape=output_shape)

# create result dir
if not os.path.exists(os.path.join(result_folder)):
    os.mkdir(result_folder)

for n in range(n_examples):

    # generate new image and corresponding labels
    im, lab = brain_generator.generate_image()

    # save output image and label map
    utils.save_volume(im, brain_generator.aff, brain_generator.header,
                      os.path.join(result_folder, 'image_%s.nii.gz' % n))
    utils.save_volume(lab, brain_generator.aff, brain_generator.header,
                      os.path.join(result_folder, 'labels_%s.nii.gz' % n))
