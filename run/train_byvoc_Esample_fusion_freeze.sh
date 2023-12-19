cd ..
python main.py --resume --freeze-head --sub-name train_byvoc_Esample_fusion --use-sample Easy --n-epoch 50 --optimizer AdamW --loss fusion --Dice-rate 2.1 --BCE-rate 1.2 --freeze-head
