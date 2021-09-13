python ./examples/run_classification.py --model_name_or_path roberta-base \
--task_name ag --max_seq_length 256 --per_device_train_batch_size 16 \
--learning_rate 4e-5 --num_train_epochs 3.0 --output_dir ./results/ag_FT/ \
--data_dir ./data/AG/ --Ngram_path ./ngram/pmi_ag_ngram.txt \
--fasttext_model_path ./ngram/ag.npy --overwrite_output_dir
