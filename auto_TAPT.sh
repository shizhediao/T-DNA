python ./examples/run_language_modeling.py \
--output_dir=./models/ag_TAPT/ --model_type=roberta  --overwrite_output_dir \
--model_name_or_path=roberta-base --train_data_file=./data/tapt_data/AG/train.tsv \
--eval_data_file=./data/tapt_data/AG/dev.tsv --mlm --line_by_line \
--Ngram_path ./ngram/pmi_ag_ngram.txt --num_train_epochs 10.0 \
--fasttext_model_path ./ngram/ag.npy --learning_rate 4e-5

python ./examples/run_classification.py \
--model_name_or_path ./models/ag_TAPT \
--task_name ag --max_seq_length 256 --per_device_train_batch_size 16 \
--learning_rate 2e-5 --num_train_epochs 5.0 --output_dir ./results/ag_TAPT_FT/ \
--data_dir ./data/AG/ --Ngram_path ./ngram/pmi_ag_ngram.txt --overwrite_output_dir --save_steps 5000