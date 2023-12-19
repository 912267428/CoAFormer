cd ..
python main.py --resume --sub-name train_byvoc_DCsample_fusion --use-sample Double_Crossing --n-epoch 50 --optimizer AdamW --loss fusion --Dice-rate 2.1 --BCE-rate 1.2 --freeze-head
