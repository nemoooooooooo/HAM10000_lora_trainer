import os, re
import torch
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from transformers import PretrainedConfig
from typing import List

# Modified from https://github.com/replicate/cog-sdxl/blob/main/dataset_and_utils.py
class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.train_ids_t5: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_toks: List[str]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_toks, list), "inserting_toks should be a list of strings."
            assert all(isinstance(tok, str) for tok in inserting_toks), (
                "All elements in inserting_toks should be strings."
            )

            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Convert the token abstractions to ids
            if idx == 0:
                self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
            else:
                self.train_ids_t5 = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            embeds = (
                text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.encoder.embed_tokens
            )
            std_token_embedding = embeds.weight.data.std()

            logger.info(f"{idx} text encoder's std_token_embedding: {std_token_embedding}")

            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            # if initializer_concept are not provided, token embeddings are initialized randomly
            if args.initializer_concept is None:
                hidden_size = (
                    text_encoder.text_model.config.hidden_size if idx == 0 else text_encoder.encoder.config.hidden_size
                )
                embeds.weight.data[train_ids] = (
                    torch.randn(len(train_ids), hidden_size).to(device=self.device).to(dtype=self.dtype)
                    * std_token_embedding
                )
            else:
                # Convert the initializer_token, placeholder_token to ids
                initializer_token_ids = tokenizer.encode(args.initializer_concept, add_special_tokens=False)
                for token_idx, token_id in enumerate(train_ids):
                    embeds.weight.data[token_id] = (embeds.weight.data)[
                        initializer_token_ids[token_idx % len(initializer_token_ids)]
                    ].clone()

            self.embeddings_settings[f"original_embeddings_{idx}"] = embeds.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            # makes sure we don't update any embedding weights besides the newly added token
            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = index_no_updates

            logger.info(self.embeddings_settings[f"index_no_updates_{idx}"].shape)

            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_one, idx==0 - CLIP ViT-L/14, text_encoder_two, idx==1 - T5 xxl
        idx_to_text_encoder_name = {0: "clip_l", 1: "t5"}
        for idx, text_encoder in enumerate(self.text_encoders):
            train_ids = self.train_ids if idx == 0 else self.train_ids_t5
            embeds = text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.shared
            assert embeds.weight.data.shape[0] == len(self.tokenizers[idx]), "Tokenizers should be the same."
            new_token_embeddings = embeds.weight.data[train_ids]

            # New tokens for each text encoder are saved under "clip_l" (for text_encoder 0),
            # Note: When loading with diffusers, any name can work - simply specify in inference
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
            # tensors[f"text_encoders_{idx}"] = new_token_embeddings

        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            embeds = text_encoder.text_model.embeddings.token_embedding if idx == 0 else text_encoder.shared
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            embeds.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = embeds.weight.data[index_updates]
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            embeds.weight.data[index_updates] = new_embeddings



def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def tokenize_prompt(tokenizer, prompt, max_sequence_length, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two

def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    train_text_encoder_ti=False,
    enable_t5_ti=False,
    pure_textual_inversion=False,
    token_abstraction_dict=None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    trigger_str = f"You should use {instance_prompt} to trigger the image generation."

    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )
    diffusers_load_lora = ""
    diffusers_imports_pivotal = ""
    diffusers_example_pivotal = ""
    if not pure_textual_inversion:
        diffusers_load_lora = (
            f"""pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')"""
        )
    if train_text_encoder_ti:
        embeddings_filename = f"{repo_folder}_emb"
        ti_keys = ", ".join(f'"{match}"' for match in re.findall(r"<s\d+>", instance_prompt))
        trigger_str = (
            "To trigger image generation of trained concept(or concepts) replace each concept identifier "
            "in you prompt with the new inserted tokens:\n"
        )
        diffusers_imports_pivotal = """from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
            """
        if enable_t5_ti:
            diffusers_example_pivotal = f"""embedding_path = hf_hub_download(repo_id='{repo_id}', filename='{embeddings_filename}.safetensors', repo_type="model")
    state_dict = load_file(embedding_path)
    pipeline.load_textual_inversion(state_dict["clip_l"], token=[{ti_keys}], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
    pipeline.load_textual_inversion(state_dict["t5"], token=[{ti_keys}], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
            """
        else:
            diffusers_example_pivotal = f"""embedding_path = hf_hub_download(repo_id='{repo_id}', filename='{embeddings_filename}.safetensors', repo_type="model")
    state_dict = load_file(embedding_path)
    pipeline.load_textual_inversion(state_dict["clip_l"], token=[{ti_keys}], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
            """
        if token_abstraction_dict:
            for key, value in token_abstraction_dict.items():
                tokens = "".join(value)
                trigger_str += f"""
    to trigger concept `{key}` → use `{tokens}` in your prompt \n
    """

    model_description = f"""
# Flux DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).

Was LoRA for the text encoder enabled? {train_text_encoder}.

Pivotal tuning was enabled: {train_text_encoder_ti}.

## Trigger words

{trigger_str}

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
{diffusers_imports_pivotal}
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
{diffusers_load_lora}
{diffusers_example_pivotal}
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


