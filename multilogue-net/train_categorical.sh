python train_categorical.py \
	--lr=1e-4 --l2=1e-5 --rec-dropout=0.1 --dropout=0.5 \
	--batch-size=128 --epochs=50 --class-weight \
	--log_dir="logs/mosei_$1" \
	--model_path="model/categorical_$1.model" > "model/categorical_$1.result"
