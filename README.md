MaskVton
Visual Try-ON

Currently, Most of VTON method focus utilize diffusion model to achieve the aim of VTON. They highly dependent on the capacity of diffusion, but ignore the reasonable of dataset. Ultimately, terrible construction of dataset cause the DM method to degrate performance.

Trouble: paired dataset utilize target person and corresponed paired garment with other human parsing results (keypoint, densepose, garment mask, image_mask and so) as model inputs. Absolute paired data make the DM severely rely on the mask of garment to generate visual try-on result. Simultaneously, this behaviour may import the model with the attributes of garment. Therefore, current methods merely can cope with the Vton task between same attribute target person garment and garment. However User constantly leverage differnt categories garment as the input, and model can't finish task at this kind of case.

Method: Introducing a exclusive mask generator and multi-value strategy to finish this task.

baseline: CAT-DM

waring:

All preparation follow the guidence of CAT-DM.


when you prepare dataset for training, you have to run tools/gt_mask_dresscode.py to generate groundtruth mask for dress code.

python tools/gt_mask_dresscode.py datasets/dresscode/dresses datasets/dresscode/dresses/gt_mask

python tools/gt_mask_dresscode.py datasets/dresscode/lower_body datasets/dresscode/lower_body/gt_mask

python tools/gt_mask_dresscode.py datasets/dresscode/upper_body datasets/dresscode/upper_body/gt_mask


In addition, after finishing one stage of training mask generator, the mask generator need to be runed with the target data in order to generate warped garment mask(test stage of mask generator). Next, position these mask to corresponding data path. Finally, you can state the train stage of vton.

**********************************************************************
Very important thing is that the programme only is a debug code to vertify mu idea whether can achieve a good result or not compared to the original method. Therefor, the programme is a temporary programme, and now it had been deserted.
***********************************************************************
