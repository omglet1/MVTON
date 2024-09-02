#该文件用于为Viton数据集上训练的掩码生成器进行测试
python -u test_mask.py \
--base configs/test_vitonhd_mask_unet.yaml \
--ckpt logs/2024-08-30T09-32-02_train_vitonhd_mask_unet/testtube/version_0/checkpoints/epoch=199-step=72799.ckpt \
--ddim 16
