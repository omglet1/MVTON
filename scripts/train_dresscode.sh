python -u train.py \
--logdir logs_dresscode_vton \
--pretrained_model checkpoints/pbe_dim5.ckpt \
--base configs/train_dresscode_mvmask.yaml


#多数值掩码
#--base configs/train_dresscode_mvmask.yaml

#正常训练
#--base configs/train_dresscode.yaml