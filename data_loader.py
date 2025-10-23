from torch.utils.data import Dataset
import itertools
from torchvision import transforms
from PIL.ImageOps import exif_transpose
from torchvision.transforms.functional import crop
import torch

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        args,
        instance_data_root,
        instance_prompt,
        class_prompt,
        train_text_encoder_ti,
        token_abstraction_dict=None,  # token mapping for textual inversion
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
    ):
        self.size = size

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode {interpolation=}.")
        
        if isinstance(self.size, tuple):
            h, w = self.size
            train_resize = transforms.Resize((h, w), interpolation=interpolation)
            train_crop   = transforms.CenterCrop((h, w)) if args.center_crop else transforms.RandomCrop((h, w))
        else:
            train_resize = transforms.Resize(self.size, interpolation=interpolation)
            train_crop   = transforms.CenterCrop(self.size) if args.center_crop else transforms.RandomCrop(self.size)

        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                image = train_crop(image)
            else:
                if isinstance(self.size, tuple):
                    y1, x1, h, w = train_crop.get_params(image, self.size)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (self.size, self.size))
                    image = crop(image, y1, x1, h, w)


            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size) if args.center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                if self.train_text_encoder_ti:
                    # replace instances of --token_abstraction in caption with the new tokens: "<si><si+1>" etc.
                    for token_abs, token_replacement in self.token_abstraction_dict.items():
                        caption = caption.replace(token_abs, "".join(token_replacement))
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # the given instance prompt is used for all images
            example["instance_prompt"] = self.instance_prompt


        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch
