cd ..
python main.py --sub-name train_byvoc_DCsample_fusion_prefusion --use-sample Double_Crossing --n-epoch 50 --optimizer AdamW --loss fusion --Dice-rate 2.1 --BCE-rate 1.2 --freeze-head --head-pretrain-dir ./Checkpoints/CoAFormer/train_byvoc2012_noUpSample_fusion/BestIou.pth
