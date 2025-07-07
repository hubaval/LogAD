import argparse
import os
from datetime import datetime

import datasets
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, logging, AutoModelForCausalLM, TrainerState
from transformers.trainer import TRAINER_STATE_NAME
from trl import SFTTrainer, SFTConfig, get_kbit_device_map
from transformers.trainer_utils import get_last_checkpoint
from accelerate import Accelerator
from py_scipts.utils import get_peft_config


def formatting_func(example):
    """
    Prompt formatting
    :param example:
    :return:
    """
    text = f"### Context: {example['input_text']}\n ### Next: {example['target_text']}"
    return text


def main():
    # Quantization configuration
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.float32,  #32 for stable mixed precision
        )

    # LoRa configuration
    peft_config = get_peft_config()

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args_pars.model,
        torch_dtype=torch.float32,
        quantization_config=nf4_config,
        attn_implementation="sdpa",
        local_files_only=True,
        device_map=get_kbit_device_map(),
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args_pars.model, local_files_only=True)  # Local file is True when no internet access
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    data = datasets.load_from_disk(args_pars.dataset)

    # Define training configuration
    args = SFTConfig(
        do_train=True,
        per_device_train_batch_size=args_pars.batch,
        per_device_eval_batch_size=args_pars.batch,
        gradient_accumulation_steps=args_pars.grad,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy="steps",
        eval_steps=50,
        fp16_full_eval=True,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        warmup_steps=10,
        num_train_epochs=1,
        report_to="wandb",
        learning_rate=5e-4,
        output_dir=output_dir,
        max_seq_length=args_pars.context,
        dataset_text_field='text',
        disable_tqdm=True,
        run_name=args_pars.rname,
        packing=True,
        )

    trainer = SFTTrainer(
        model=base_model,
        peft_config=peft_config,
        train_dataset=data['train'],
        eval_dataset=data['test'],
        args=args,
        tokenizer=tokenizer,
        formatting_func=formatting_func if "-ftd" not in args_pars.dataset else None,  # Separate domain-specific fine-tuning and task-specific fine-tuning
        )

    # Verify if checkpoint needs to be saved (due to parallelization issues)
    if args_pars.is_save_checkpoint:
        if isinstance(output_dir, bool) and output_dir:
            resume_from_checkpoint = get_last_checkpoint(output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({output_dir})")

            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            trainer._load_from_checkpoint(resume_from_checkpoint=state.best_model_checkpoint)


    else:
        trainer.accelerator.print(f"{trainer.model}")
        if getattr(trainer.accelerator.state, "fsdp_plugin", None):
            from peft.utils.other import fsdp_auto_wrap_policy

            fsdp_plugin = trainer.accelerator.state.fsdp_plugin
            fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

        trainer.accelerator.print(f"\nTraining model : {args_pars.model} on PEFT\n"
                                  f'\t"run" : "{args_pars.rname}",\n'
                                  f'\t"output_saved" : "{save_run}",\n'
                                  f'\t"dataset" : "{args_pars.dataset}",\n'
                                  f'\t\t"learning_rate" : "{args.learning_rate}",\n'
                                  f'\t\t"context" : "{args_pars.context}",\n'
                                  f'\t\t"batch" : "{args_pars.batch}",\n'
                                  f'\t\t"is_checkpoint" : "{args_pars.checkpoint}",\n'
                                  f'\t\t"checkpoint_path" : "{output_dir}",\n'
                                  f'\t"start_time" : "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"\n\n')

        trainer.train(resume_from_checkpoint=args_pars.checkpoint)

    # Verify parallelization is enabled
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(save_run)


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action='store', type=str, required=True)
    parser.add_argument("--run-name", action='store', type=str, required=True, dest='rname')
    parser.add_argument("--dataset", action="store", type=str, help="dataset path")
    parser.add_argument("--batch", action="store", type=int, help="batch size", default=1)
    parser.add_argument("--grad", action="store", type=int, help="gradient accumulation step", default=4)
    parser.add_argument("--context", action="store", type=int, help="context size (input)", default=2048)
    parser.add_argument("--root", action="store", type=str, help="root path (default ./run)", default="./run")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, help="Resume from checkpoint or not")
    parser.add_argument("--save_checkpoint", action=argparse.BooleanOptionalAction, help="Wether to save fsdp peft training from existing checkpoint", dest="is_save_checkpoint")
    return parser


if __name__ == "__main__":
    args_pars = setup_parser().parse_args()

    run_path = f'{args_pars.root}/{args_pars.rname}'
    save_run = f'{run_path}/model/'
    output_dir = f'{run_path}/checkpoint/'
    run_name = args_pars.rname

    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        'LogAD',
        init_kwargs={
            "wandb": {
                "mode": "offline",
                'dir': f'{run_path}/',
                "resume": "auto" if args_pars.checkpoint else None,
                "name": run_name
                }
            }
        )

    logging.disable_progress_bar()
    datasets.disable_progress_bars()

    main()

    accelerator.end_training()
