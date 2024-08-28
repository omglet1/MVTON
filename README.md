MaskVton
Visual Try-ON

Currently, Most of VTON method focus utilize diffusion model to achieve the aim of VTON. They highly dependent on the capacity of diffusion, but ignore the reasonable of dataset. Ultimately, terrible construction of dataset cause the DM method to degrate performance.

Trouble: paired dataset utilize target person and corresponed paired garment with other human parsing results (keypoint, densepose, garment mask, image_mask and so) as model inputs. Absolute paired data make the DM severely rely on the mask of garment to generate visual try-on result. Simultaneously, this behaviour may import the model with the attributes of garment. Therefore, current methods merely can cope with the Vton task between same attribute target person garment and garment. However User constantly leverage differnt categories garment as the input, and model can't finish task at this kind of case.

Method: Introducing a exclusive mask generator and multi-value strategy to finish this task.

baseline: CAT-DM