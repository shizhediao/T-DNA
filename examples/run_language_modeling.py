import logging
import math
import os
from typing import List
import torch
import sys

from os.path import dirname, abspath
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(1,parent_dir+"/TDNA")

from configuration import AutoConfig
from trainer import Trainer, set_seed
from util import DataCollatorForLanguageModeling, LineByLineTextDataset
from tokenization import RobertaTokenizer, PreTrainedTokenizer
from modeling import RobertaForMaskedLM


import argparse
import numpy as np
from enum import Enum
import socket
from datetime import datetime

logger = logging.getLogger(__name__)

class EvaluationStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"

class TDNANgramDict(object):
    """
    Dict class to store the ngram
    """

    def __init__(self, ngram_freq_path, max_ngram_in_seq=20):
        """Constructs TDNANgramDict

        :param ngram_freq_path: ngrams with frequency
        """
        self.ngram_freq_path = ngram_freq_path
        self.max_ngram_in_seq = max_ngram_in_seq
        self.id_to_ngram_list = []
        self.ngram_to_id_dict = {}

        logger.info("loading ngram frequency file {}".format(ngram_freq_path))
        with open(ngram_freq_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                ngram = line.strip()
                self.id_to_ngram_list.append(ngram)
                self.ngram_to_id_dict[ngram] = i

    def save(self, ngram_freq_path):
        with open(ngram_freq_path, "w", encoding="utf-8") as fout:
            for ngram, freq in self.ngram_to_freq_dict.items():
                fout.write("{},{}\n".format(ngram, freq))


def get_dataset(
        args,
        Ngram_dict,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool = False,
        cache_dir=None,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return LineByLineTextDataset(tokenizer=tokenizer, Ngram_dict=Ngram_dict, file_path=file_path,
                                 block_size=args.block_size)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--fasttext_model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pretrained fastText model for initializing ngram embeddings")
    parser.add_argument("--Ngram_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to Ngram path")
    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        help="If training from scratch, pass a model type")
    parser.add_argument("--config_name",
                        default=None,
                        type=str,
                        required=False,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",
                        default=None,
                        type=str,
                        required=False,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="Where do you want to store the pretrained models downloaded from s3")
    parser.add_argument("--train_data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--eval_data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--line_by_line",
                        action='store_true',
                        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.")
    parser.add_argument("--mlm",
                        action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability",
                        default=0.15,
                        type=float,
                        required=False,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--plm_probability",
                        default=1 / 6,
                        type=float,
                        required=False,
                        help="Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling.")
    parser.add_argument("--max_span_length",
                        default=5,
                        type=int,
                        required=False,
                        help="Maximum length of a span of masked tokens for permutation language modeling.")
    parser.add_argument("--block_size",
                        default=-1,
                        type=int,
                        required=False,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--overwrite_cache",
                        action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--overwrite_output_dir",
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--do_train",
                        action='store_false',
                        help="Whether to run training")
    parser.add_argument("--do_eval",
                        action='store_false',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_false',
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",
                        action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--evaluation_strategy",
                        default="no",
                        type=EvaluationStrategy,
                        required=False,
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--prediction_loss_only",
                        action='store_true',
                        help="When performing evaluation and predictions, only returns the loss.")
    parser.add_argument("--per_device_train_batch_size",
                        default=8,
                        type=int,
                        required=False,
                        help="Batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size",
                        default=8,
                        type=int,
                        required=False,
                        help="Batch size per GPU/TPU core/CPU for evaluation.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=None,
                        type=int,
                        required=False,
                        help="Deprecated, the use of `--per_device_train_batch_size` is preferred. ")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=None,
                        type=int,
                        required=False,
                        help="Deprecated, the use of `--per_device_eval_batch_size` is preferred.")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        required=False,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        required=False,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        required=False,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_beta1",
                        default=0.9,
                        type=float,
                        required=False,
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--adam_beta2",
                        default=0.999,
                        type=float,
                        required=False,
                        help="Beta2 for Adam optimizer")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        required=False,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        required=False,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        required=False,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1,
                        type=int,
                        required=False,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        required=False,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_dir",
                        default="runs/" + datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname(),
                        type=str,
                        required=False,
                        help="Tensorboard log dir.")
    parser.add_argument("--logging_first_step",
                        action='store_true',
                        help="Log and eval the first global_step")
    parser.add_argument("--logging_steps",
                        default=500,
                        type=int,
                        required=False,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        default=500000,
                        type=int,
                        required=False,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit",
                        default=None,
                        type=int,
                        required=False,
                        help="Limit the total amount of checkpoints.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Do not use CUDA even when it is available")
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        required=False,
                        help="random seed for initialization")
    parser.add_argument("--fp16",
                        action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",
                        default="O1",
                        type=str,
                        required=False,
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        required=False,
                        help="For distributed training: local_rank")
    parser.add_argument("--tpu_num_cores",
                        default=None,
                        type=int,
                        required=False,
                        help="TPU: Number of TPU cores (automatically passed by launcher script)")
    parser.add_argument("--tpu_metrics_debug",
                        action='store_true',
                        help="Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics")
    parser.add_argument("--debug",
                        action='store_true',
                        help="Whether to print debug metrics on TPU")
    parser.add_argument("--dataloader_drop_last",
                        action='store_true',
                        help="Drop the last incomplete batch if it is not divisible by the batch size.")
    parser.add_argument("--eval_steps",
                        default=None,
                        type=int,
                        required=False,
                        help="Run an evaluation every X steps.")
    parser.add_argument("--dataloader_num_workers",
                        default=0,
                        type=int,
                        required=False,
                        help="Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process.")
    parser.add_argument("--past_index",
                        default=-1,
                        type=int,
                        required=False,
                        help="If >=0, uses the corresponding part of the output as the past state for next step.")
    parser.add_argument("--run_name",
                        default=None,
                        type=str,
                        required=False,
                        help="An optional descriptor for the run. Notably used for wandb logging.")
    parser.add_argument("--disable_tqdm",
                        action='store_true',
                        help="Whether or not to disable the tqdm progress bars.")
    parser.add_argument("--remove_unused_columns",
                        action='store_false',
                        help="Remove columns not required by the model when using an nlp.Dataset.")
    parser.add_argument("--label_names",
                        default=None,
                        type=List[str],
                        required=False,
                        help="The list of keys in your dictionary of inputs that correspond to the label")

    args = parser.parse_args()

    if args.no_cuda:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    elif args.local_rank == -1:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.distributed.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1

    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    Ngram_dict = TDNANgramDict(args.Ngram_path)

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir,
                                            Ngram_vocab_size=len(Ngram_dict.id_to_ngram_list))

    if args.tokenizer_name:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir,
                                                     Ngram_vocab_size=len(Ngram_dict.id_to_ngram_list))
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    model = RobertaForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    pretrained_embedding_np = np.load(args.fasttext_model_path)
    pretrained_embedding = torch.from_numpy(pretrained_embedding_np)
    model.roberta.Ngram_embeddings.word_embeddings.weight.data.copy_(pretrained_embedding)

    # Get datasets

    train_dataset = (
        get_dataset(args, Ngram_dict, tokenizer=tokenizer, cache_dir=args.cache_dir) if args.do_train else None
    )
    eval_dataset = (
        get_dataset(args, Ngram_dict, tokenizer=tokenizer, evaluate=True, cache_dir=args.cache_dir)
        if args.do_eval
        else None
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=args.mlm, mlm_probability=args.mlm_probability
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if args.do_train:
        model_path = (
            args.model_name_or_path
            if args.model_name_or_path is not None and os.path.isdir(args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(args.output_dir)

    # Evaluation
    results = {}
    if args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
