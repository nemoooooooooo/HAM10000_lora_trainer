import copy
import itertools
import logging
import math
import os
import random
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional
from args_parser import args
from data_loader import DreamBoothDataset, collate_fn

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb


from utils import import_model_class_from_model_name_or_path, load_text_encoders, encode_prompt, TokenEmbeddingsHandler

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)




def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,  # expects {"prompts": List[str]}
    step,
    torch_dtype,
    is_final_validation=False,
):
    prompts: List[str] = pipeline_args.get("prompts", [])
    if isinstance(prompts, str):
        prompts = [prompts]

    if not prompts:
        logger.info("No validation prompts provided; skipping validation.")
        return []

    logger.info(
        f"Running validation at step {step}...\n"
        f"Generating {args.num_validation_images} images for each of {len(prompts)} prompt(s)."
    )

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    all_images = []
    for pi, prompt in enumerate(prompts):
        # pre-encode once per prompt to keep behavior with T5 autocast constraints
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(prompt, prompt_2=prompt)

        prompt_images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                img = pipeline(
                    height=1184, width=1728,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    generator=generator,
                ).images[0]
                prompt_images.append(img)
                all_images.append(img)

        # Log per-prompt to trackers
        for tracker in accelerator.trackers:
            phase_name = "test" if is_final_validation else "validation"
            caption_prefix = f"{prompt}"
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(im) for im in prompt_images])
                # add per-prompt channel
                tracker.writer.add_images(f"{phase_name}/{pi}", np_images, step, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        phase_name: [
                            wandb.Image(im, caption=f"{caption_prefix}") for i, im in enumerate(prompt_images)
                        ]
                    }
                )

    del pipeline
    free_memory()
    return all_images




def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        model_id = args.hub_model_id or Path(args.output_dir).name
        repo_id = None
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=model_id,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    if args.train_text_encoder_ti:
        # we parse the provided token identifier (or identifiers) into a list. s.t. - "TOK" -> ["TOK"], "TOK,
        # TOK2" -> ["TOK", "TOK2"] etc.
        token_abstraction_list = [place_holder.strip() for place_holder in re.split(r",\s*", args.token_abstraction)]
        logger.info(f"list of token identifiers: {token_abstraction_list}")

        if args.initializer_concept is None:
            num_new_tokens_per_abstraction = (
                2 if args.num_new_tokens_per_abstraction is None else args.num_new_tokens_per_abstraction
            )
        # if args.initializer_concept is provided, we ignore args.num_new_tokens_per_abstraction
        else:
            token_ids = tokenizer_one.encode(args.initializer_concept, add_special_tokens=False)
            num_new_tokens_per_abstraction = len(token_ids)
            if args.enable_t5_ti:
                token_ids_t5 = tokenizer_two.encode(args.initializer_concept, add_special_tokens=False)
                num_new_tokens_per_abstraction = max(len(token_ids), len(token_ids_t5))
            logger.info(
                f"initializer_concept: {args.initializer_concept}, num_new_tokens_per_abstraction: {num_new_tokens_per_abstraction}"
            )

        token_abstraction_dict = {}
        token_idx = 0
        for i, token in enumerate(token_abstraction_list):
            token_abstraction_dict[token] = [f"<s{token_idx + i + j}>" for j in range(num_new_tokens_per_abstraction)]
            token_idx += num_new_tokens_per_abstraction - 1

        # replace instances of --token_abstraction in --instance_prompt with the new tokens: "<si><si+1>" etc.
        for token_abs, token_replacement in token_abstraction_dict.items():
            new_instance_prompt = args.instance_prompt.replace(token_abs, "".join(token_replacement))
            if args.instance_prompt == new_instance_prompt:
                logger.warning(
                    "Note! the instance prompt provided in --instance_prompt does not include the token abstraction specified "
                    "--token_abstraction. This may lead to incorrect optimization of text embeddings during pivotal tuning"
                )
            args.instance_prompt = new_instance_prompt
            if args.with_prior_preservation:
                args.class_prompt = args.class_prompt.replace(token_abs, "".join(token_replacement))
            if args.validation_prompt:
                args.validation_prompt = args.validation_prompt.replace(token_abs, "".join(token_replacement))

        # initialize the new tokens for textual inversion
        text_encoders = [text_encoder_one, text_encoder_two] if args.enable_t5_ti else [text_encoder_one]
        tokenizers = [tokenizer_one, tokenizer_two] if args.enable_t5_ti else [tokenizer_one]
        embedding_handler = TokenEmbeddingsHandler(text_encoders, tokenizers)
        inserting_toks = []
        for new_tok in token_abstraction_dict.values():
            inserting_toks.extend(new_tok)
        embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    if args.train_text_encoder:  # when --train_text_encoder_ti we don't save the layers
                        text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                        modules_to_save["text_encoder"] = model
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    pass  # when --train_text_encoder_ti and --enable_t5_ti we don't save the layers
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )
        if args.train_text_encoder_ti:
            embedding_handler.save_embeddings(f"{args.output_dir}/{Path(args.output_dir).name}_emb.safetensors")

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
    # if we use textual inversion, we freeze all parameters except for the token embeddings
    # in text encoder
    elif args.train_text_encoder_ti:
        text_lora_parameters_one = []  # CLIP
        for name, param in text_encoder_one.named_parameters():
            if "token_embedding" in name:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                if args.mixed_precision == "fp16":
                    param.data = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_one.append(param)
            else:
                param.requires_grad = False
        if args.enable_t5_ti:  # whether to do pivotal tuning/textual inversion for T5 as well
            text_lora_parameters_two = []
            for name, param in text_encoder_two.named_parameters():
                if "shared" in name:
                    # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                    if args.mixed_precision == "fp16":
                        param.data = param.to(dtype=torch.float32)
                    param.requires_grad = True
                    text_lora_parameters_two.append(param)
                else:
                    param.requires_grad = False

    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)

    # if --train_text_encoder_ti and train_transformer_frac == 0 where essentially performing textual inversion
    # and not training transformer LoRA layers
    pure_textual_inversion = args.train_text_encoder_ti and args.train_transformer_frac == 0

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    if not freeze_text_encoder:
        # different learning rate for text encoder and transformer
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": args.text_encoder_lr,
        }
        if not args.enable_t5_ti:
            # pure textual inversion - only clip
            if pure_textual_inversion:
                params_to_optimize = [text_parameters_one_with_lr]
                te_idx = 0
            else:  # regular te training or regular pivotal for clip
                params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]
                te_idx = 1
        elif args.enable_t5_ti:
            # pivotal tuning of clip & t5
            text_parameters_two_with_lr = {
                "params": text_lora_parameters_two,
                "weight_decay": args.adam_weight_decay_text_encoder
                if args.adam_weight_decay_text_encoder
                else args.adam_weight_decay,
                "lr": args.text_encoder_lr,
            }
            # pure textual inversion - only clip & t5
            if pure_textual_inversion:
                params_to_optimize = [text_parameters_one_with_lr, text_parameters_two_with_lr]
                te_idx = 0
            else:  # regular pivotal tuning of clip & t5
                params_to_optimize = [
                    transformer_parameters_with_lr,
                    text_parameters_one_with_lr,
                    text_parameters_two_with_lr,
                ]
                te_idx = 1
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if not freeze_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters to be
            # --learning_rate

            params_to_optimize[te_idx]["lr"] = args.learning_rate
            params_to_optimize[-1]["lr"] = args.learning_rate
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        args=args,
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        train_text_encoder_ti=args.train_text_encoder_ti,
        token_abstraction_dict=token_abstraction_dict if args.train_text_encoder_ti else None,
        class_prompt=args.class_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    if freeze_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if freeze_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders, text_encoder_one, text_encoder_two
        free_memory()

    # if --train_text_encoder_ti we need add_special_tokens to be True for textual inversion
    add_special_tokens_clip = True if args.train_text_encoder_ti else False
    add_special_tokens_t5 = True if (args.train_text_encoder_ti and args.enable_t5_ti) else False

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if freeze_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(
                tokenizer_one, args.instance_prompt, max_sequence_length=77, add_special_tokens=add_special_tokens_clip
            )
            tokens_two = tokenize_prompt(
                tokenizer_two,
                args.instance_prompt,
                max_sequence_length=args.max_sequence_length,
                add_special_tokens=add_special_tokens_t5,
            )
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(
                    tokenizer_one,
                    args.class_prompt,
                    max_sequence_length=77,
                    add_special_tokens=add_special_tokens_clip,
                )
                class_tokens_two = tokenize_prompt(
                    tokenizer_two,
                    args.class_prompt,
                    max_sequence_length=args.max_sequence_length,
                    add_special_tokens=add_special_tokens_t5,
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

        if args.validation_prompt is None:
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if not freeze_text_encoder:
        if args.enable_t5_ti:
            (
                transformer,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = accelerator.prepare(
                transformer,
                text_encoder_one,
                text_encoder_two,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )
        else:
            transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer, text_encoder_one, optimizer, train_dataloader, lr_scheduler
            )

    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-dev-lora-advanced"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    if args.train_text_encoder:
        num_train_epochs_text_encoder = int(args.train_text_encoder_frac * args.num_train_epochs)
        num_train_epochs_transformer = int(args.train_transformer_frac * args.num_train_epochs)
    elif args.train_text_encoder_ti:  # args.train_text_encoder_ti
        num_train_epochs_text_encoder = int(args.train_text_encoder_ti_frac * args.num_train_epochs)
        num_train_epochs_transformer = int(args.train_transformer_frac * args.num_train_epochs)

    # flag used for textual inversion
    pivoted_te = False
    pivoted_tr = False
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        # if performing any kind of optimization of text_encoder params
        if args.train_text_encoder or args.train_text_encoder_ti:
            if epoch == num_train_epochs_text_encoder:
                # flag to stop text encoder optimization
                logger.info(f"PIVOT TE {epoch}")
                pivoted_te = True
            else:
                # still optimizing the text encoder
                if args.train_text_encoder:
                    text_encoder_one.train()
                    # set top parameter requires_grad = True for gradient checkpointing works
                    unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
                elif args.train_text_encoder_ti:  # textual inversion / pivotal tuning
                    text_encoder_one.train()
                if args.enable_t5_ti:
                    text_encoder_two.train()

            if epoch == num_train_epochs_transformer:
                # flag to stop transformer optimization
                logger.info(f"PIVOT TRANSFORMER {epoch}")
                pivoted_tr = True

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if not freeze_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
                if args.enable_t5_ti:
                    models_to_accumulate.extend([text_encoder_two])
            if pivoted_te:
                # stopping optimization of text_encoder params
                optimizer.param_groups[te_idx]["lr"] = 0.0
                optimizer.param_groups[-1]["lr"] = 0.0
            elif pivoted_tr and not pure_textual_inversion:
                logger.info(f"PIVOT TRANSFORMER {epoch}")
                optimizer.param_groups[0]["lr"] = 0.0

            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    elems_to_repeat = 1
                    if freeze_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(
                            tokenizer_one, prompts, max_sequence_length=77, add_special_tokens=add_special_tokens_clip
                        )
                        tokens_two = tokenize_prompt(
                            tokenizer_two,
                            prompts,
                            max_sequence_length=args.max_sequence_length,
                            add_special_tokens=add_special_tokens_t5,
                        )
                else:
                    elems_to_repeat = len(prompts)

                if not freeze_text_encoder:
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[None, None],
                        text_input_ids_list=[
                            tokens_one.repeat(elems_to_repeat, 1),
                            tokens_two.repeat(elems_to_repeat, 1),
                        ],
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                        prompt=prompts,
                    )
                # Convert images to latent space
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not freeze_text_encoder:
                        if args.train_text_encoder:  # text encoder tuning
                            params_to_clip = itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        elif pure_textual_inversion:
                            if args.enable_t5_ti:
                                params_to_clip = itertools.chain(
                                    text_encoder_one.parameters(), text_encoder_two.parameters()
                                )
                            else:
                                params_to_clip = itertools.chain(text_encoder_one.parameters())
                        else:
                            if args.enable_t5_ti:
                                params_to_clip = itertools.chain(
                                    transformer.parameters(),
                                    text_encoder_one.parameters(),
                                    text_encoder_two.parameters(),
                                )
                            else:
                                params_to_clip = itertools.chain(
                                    transformer.parameters(), text_encoder_one.parameters()
                                )
                    else:
                        params_to_clip = itertools.chain(transformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # every step, we reset the embeddings to the original embeddings.
                if args.train_text_encoder_ti:
                    embedding_handler.retract_embeddings()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if args.train_text_encoder_ti:
                            embedding_handler.save_embeddings(
                                f"{args.output_dir}/{Path(args.output_dir).name}_emb_checkpoint_{global_step}.safetensors"
                            )
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        # Create pipeline
                        if freeze_text_encoder:
                            text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
                            text_encoder_one.to(weight_dtype)
                            text_encoder_two.to(weight_dtype)
                        
                        pipeline = FluxPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            vae=vae,
                            text_encoder=unwrap_model(text_encoder_one),
                            text_encoder_2=unwrap_model(text_encoder_two),
                            transformer=unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                        )
                        pipeline_args = {"prompts": args.validation_prompts}
                        images = log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            step=global_step,  # You might want to pass global_step instead
                            torch_dtype=weight_dtype,
                        )
                        if freeze_text_encoder:
                            del text_encoder_one, text_encoder_two
                            free_memory()
                        
                        del pipeline
                        images = None
                    

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            modules_to_save["text_encoder"] = text_encoder_one
        else:
            text_encoder_lora_layers = None

        if not pure_textual_inversion:
            FluxPipeline.save_lora_weights(
                save_directory=args.output_dir,
                transformer_lora_layers=transformer_lora_layers,
                text_encoder_lora_layers=text_encoder_lora_layers,
                **_collate_lora_metadata(modules_to_save),
            )

        if args.train_text_encoder_ti:
            embeddings_path = f"{args.output_dir}/{os.path.basename(args.output_dir)}_emb.safetensors"
            embedding_handler.save_embeddings(embeddings_path)

        # Final inference
        # Load previous pipeline
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        if not pure_textual_inversion:
            # load attention processors
            pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompts": args.validation_prompts}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                step=global_step,
                is_final_validation=True,
                torch_dtype=weight_dtype,
            )

        save_model_card(
            model_id if not args.push_to_hub else repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            train_text_encoder=args.train_text_encoder,
            train_text_encoder_ti=args.train_text_encoder_ti,
            enable_t5_ti=args.enable_t5_ti,
            pure_textual_inversion=pure_textual_inversion,
            token_abstraction_dict=train_dataset.token_abstraction_dict,
            instance_prompt=args.instance_prompt,
            validation_prompt=args.validation_prompt,
            repo_folder=args.output_dir,
        )
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    main(args)