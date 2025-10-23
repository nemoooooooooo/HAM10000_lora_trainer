from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
import json
from pathlib import Path
from tqdm import tqdm
import random

class HFDatasetCreator:
    def __init__(self, base_dir="ham10000_mel_flux"):
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / "train"
        self.metadata_dir = self.base_dir / "metadata"
        self.validation_dir = self.base_dir / "validation"
        self.output_dir = self.base_dir / "hf_dataset"
        self.output_dir.mkdir(exist_ok=True)
        
    def create_hf_dataset(self, 
                         use_caption_variations=True, 
                         caption_probability=0.8,
                         max_samples=None,  # Add parameter to limit samples
                         random_seed=42):    # Add seed for reproducibility
        """
        Create HuggingFace dataset from processed data
        
        Args:
            use_caption_variations: If True, randomly select from all caption variations
            caption_probability: Probability of using caption vs just trigger word (for regularization)
            max_samples: Maximum number of samples to include (None for all)
            random_seed: Random seed for reproducible sampling
        """
        print("\n🤗 Creating HuggingFace Dataset...")
        
        # Load metadata
        metadata_path = self.metadata_dir / 'dataset_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"📊 Total available samples in metadata: {len(metadata)}")
        
        # Sample if max_samples is specified
        if max_samples and max_samples < len(metadata):
            print(f"📌 Randomly sampling {max_samples} samples from {len(metadata)} available...")
            random.seed(random_seed)
            metadata = random.sample(metadata, max_samples)
            print(f"✅ Sampled {len(metadata)} samples")
        
        # Prepare training data
        train_data = {
            'image': [],
            'text': [],
            'class_label': [],
            'trigger_word': []
        }
        
        print(f"Processing {len(metadata)} training samples...")
        
        missing_images = 0
        for entry in tqdm(metadata, desc="Loading training data"):
            image_path = self.train_dir / entry['file_name']
            
            if not image_path.exists():
                print(f"⚠️ Image not found: {image_path}")
                missing_images += 1
                continue
            
            # Select caption
            if use_caption_variations and 'all_captions' in entry:
                # Randomly select from all caption variations
                caption = random.choice(entry['all_captions'])
            else:
                caption = entry['primary_caption']
            
            # Optional: Add caption dropout for regularization
            if caption_probability < 1.0:
                if random.random() > caption_probability:
                    # Use only trigger word (helps prevent overfitting)
                    caption = entry['trigger_word']
            
            train_data['image'].append(str(image_path))
            train_data['text'].append(caption)
            train_data['class_label'].append(entry['class'])
            train_data['trigger_word'].append(entry['trigger_word'])
        
        if missing_images > 0:
            print(f"⚠️ {missing_images} images were not found and skipped")
        
        print(f"✅ Successfully loaded {len(train_data['image'])} images")
        
        # Create validation data if exists
        val_data = None
        val_metadata_path = self.validation_dir / 'validation_metadata.json'
        
        if val_metadata_path.exists():
            print("\n📦 Processing validation data...")
            with open(val_metadata_path, 'r') as f:
                val_metadata = json.load(f)
            
            val_data = {
                'image': [],
                'text': [],
                'class_label': [],
                'trigger_word': []
            }
            
            for entry in val_metadata:
                val_image_path = self.validation_dir / entry['file_name']
                if val_image_path.exists():
                    val_data['image'].append(str(val_image_path))
                    val_data['text'].append(entry['caption'])
                    val_data['class_label'].append(entry['class'])
                    val_data['trigger_word'].append(entry['trigger_word'])
        else:
            print("ℹ️ No validation metadata found, creating validation split from training data...")
            # Create a small validation set from training data (1% split)
            val_data = self._create_validation_split(train_data, split_ratio=0.01)
        
        # Define features
        features = Features({
            'image': HFImage(),
            'text': Value('string'),
            'class_label': Value('string'),
            'trigger_word': Value('string')
        })
        
        # Create datasets
        print("\n📊 Creating dataset splits...")
        train_dataset = Dataset.from_dict(train_data, features=features)
        
        if val_data and len(val_data['image']) > 0:
            val_dataset = Dataset.from_dict(val_data, features=features)
            
            # Create DatasetDict with train and validation splits
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset
            })
            
            print(f"✅ Train samples: {len(train_dataset)}")
            print(f"✅ Validation samples: {len(val_dataset)}")
        else:
            dataset_dict = DatasetDict({
                'train': train_dataset
            })
            print(f"✅ Train samples: {len(train_dataset)}")
        
        # Save dataset
        dataset_path = self.output_dir / 'dataset'
        print(f"\n💾 Saving dataset to {dataset_path}...")
        dataset_dict.save_to_disk(str(dataset_path))
        
        # Also save as parquet for easier sharing
        parquet_path = self.output_dir / 'dataset.parquet'
        train_dataset.to_parquet(str(parquet_path))
        print(f"💾 Saved parquet version to {parquet_path}")
        
        # Create and save dataset info
        self._save_dataset_info(dataset_dict)
        
        # Create validation prompts file
        self._create_validation_prompts_file()
        
        print("\n✨ Dataset creation complete!")
        print(f"📂 Dataset saved to: {dataset_path}")
        
        return dataset_dict
    
    def _create_validation_split(self, train_data, split_ratio=0.01):
        """Create validation split from training data"""
        n_samples = len(train_data['image'])
        n_val = max(1, int(n_samples * split_ratio))  # Ensure at least 1 validation sample
        
        indices = list(range(n_samples))
        random.shuffle(indices)
        val_indices = set(indices[:n_val])
        
        val_data = {
            'image': [],
            'text': [],
            'class_label': [],
            'trigger_word': []
        }
        
        new_train_data = {key: [] for key in train_data.keys()}
        
        for i in range(n_samples):
            if i in val_indices:
                val_data['image'].append(train_data['image'][i])
                val_data['text'].append(train_data['text'][i])
                val_data['class_label'].append(train_data['class_label'][i])
                val_data['trigger_word'].append(train_data['trigger_word'][i])
            else:
                for key in train_data.keys():
                    new_train_data[key].append(train_data[key][i])
        
        # Update train_data
        for key in train_data.keys():
            train_data[key] = new_train_data[key]
        
        return val_data
    
    def _save_dataset_info(self, dataset_dict):
        """Save dataset information and statistics"""
        info = {
            'dataset_name': 'HAM10000 Multi-Class Dermoscopy',
            'num_train_samples': len(dataset_dict['train']) if 'train' in dataset_dict else 0,
            'num_val_samples': len(dataset_dict['validation']) if 'validation' in dataset_dict else 0,
            'classes': {},
            'trigger_words': {},
        }
        
        # Collect statistics
        if 'train' in dataset_dict:
            train_data = dataset_dict['train']
            
            # Class distribution
            from collections import Counter
            class_counts = Counter(train_data['class_label'])
            trigger_counts = Counter(train_data['trigger_word'])
            
            info['classes'] = dict(class_counts)
            info['trigger_words'] = dict(trigger_counts)
        
        # Save info
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total training samples: {info['num_train_samples']}")
        print(f"  Total validation samples: {info['num_val_samples']}")
        print(f"\n  Class distribution:")
        for cls, count in info['classes'].items():
            print(f"    {cls}: {count}")
        print(f"\n  Trigger words:")
        for trigger, count in info['trigger_words'].items():
            print(f"    {trigger}: {count}")
    
    def _create_validation_prompts_file(self):
        """Create validation prompts for monitoring training"""
        
        # Define trigger words and their variations
        validation_prompts = [
            # Melanocytic nevi
            "dermoscopic image of melnevi",
            "clinical dermoscopy showing melnevi",
            "high-resolution dermoscopy: melnevi lesion",
        ]
        
        # Save prompts
        prompts_path = self.output_dir / 'validation_prompts.txt'
        with open(prompts_path, 'w') as f:
            for prompt in validation_prompts:
                f.write(prompt + '\n')
        
        print(f"📝 Created {len(validation_prompts)} validation prompts")
        
        # Also save as JSON for more structured access
        prompts_json_path = self.output_dir / 'validation_prompts.json'
        with open(prompts_json_path, 'w') as f:
            json.dump({
                'prompts': validation_prompts,
                'trigger_words': {
                    'melnevi': 'Melanocytic nevi'
                }
            }, f, indent=2)


if __name__ == "__main__":
    # Upload ALL processed images
    creator_all = HFDatasetCreator("ham10000_mel_flux")
    dataset_dict_all = creator_all.create_hf_dataset(
        use_caption_variations=True,
        caption_probability=1,
        max_samples=None  # Upload all available samples
    )
    
    # Push to your existing repository
    repo_id = "BoooomNing/ham10000-mel-flux"
    dataset_dict_all.push_to_hub(
        repo_id,
        private=True,  # Keep it private as you had it
        max_shard_size="1GB",  # Shard large datasets
        commit_message="Upload full dataset with all samples"
    )
    print(f"✅ Pushed dataset to https://huggingface.co/datasets/{repo_id}")

