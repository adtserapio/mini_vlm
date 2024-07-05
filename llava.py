import os
import json
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import evaluate
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
from torchvision.transforms import v2
import base64
import io
from PIL import Image
import uuid

from dotenv import load_dotenv 
load_dotenv()

IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"

class VLM(pl.LightningModule):
    def __init__(self, config):
        super(VLM, self).__init__()
        self.config = config
        self.model_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            token=self.config.token
        )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.add_tokens([IMAGE_TOKEN], special_tokens=True)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        self.language_model = AutoModelForCausalLM.from_pretrained(
            self.config.language_model_name,
            token=self.config.token
        ).to(self.model_dtype)
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        
        if 'clip' in self.config.vision_model_name or 'siglip' in self.config.vision_model_name:
            clip_model = AutoModel.from_pretrained(
                self.config.vision_model_name,
                torch_dtype=self.model_dtype
            )
            self.vision_model = clip_model.vision_model
        else:
            self.vision_model = AutoModel.from_pretrained(
                self.config.vision_model_name,
                torch_dtype=self.model_dtype
            )
        self.config.num_img_patches = self.get_num_img_patches()

        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False

        self.mm_projector = nn.Linear(
            self.vision_model.config.hidden_size, 
            self.language_model.config.hidden_size,
            dtype=self.model_dtype
        )

        self.validation_results = []
        self.rouge = evaluate.load('rouge')


    def encode_images(self, pixel_values):
        device = self.language_model.device
        pixel_values = pixel_values.to(device).to(self.language_model.dtype)
        last_hidden_state = self.vision_model(pixel_values).last_hidden_state
        image_embeds = self.mm_projector(last_hidden_state)
        return image_embeds

    def prepare_for_multimodal(self, batch):
        device = self.language_model.device
        pixel_values = batch["pixel_values"]
        texts = batch["text"]

        inputs_embeds = []
        labels = []
        attention_mask = []

        image_embeds = self.encode_images(pixel_values)

        for i in range(len(texts)):
            tokenizer_outputs = self.tokenizer(
                texts[i],
                return_tensors="pt",
                padding="max_length",
                max_length=self.config.max_length,
                truncation=True
            ).to(device)
            input_ids = tokenizer_outputs.input_ids.squeeze(0)
            split_sizes = self.compute_split_sizes_from_input_ids(input_ids, self.image_token_id)
            input_ids_parts = torch.split(input_ids, split_sizes)
            before_img_input_ids, after_img_input_ids = input_ids_parts[0], input_ids_parts[1][1:]
            before_img_input_embeds = self.language_model.model.embed_tokens(before_img_input_ids)
            after_img_input_embeds = self.language_model.model.embed_tokens(after_img_input_ids)
            multimodal_embeds = torch.cat([before_img_input_embeds, image_embeds[i], after_img_input_embeds])

            ignore_tokens = torch.full((self.config.num_img_patches,), IGNORE_INDEX).to(device)
            multimodal_labels = torch.cat([before_img_input_ids, ignore_tokens, after_img_input_ids])
            multimodal_labels[multimodal_labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

            text_attention_mask = tokenizer_outputs.attention_mask.squeeze(0)
            img_attention_mask = torch.ones((self.config.num_img_patches - 1,)).to(device)
            multimodal_attention_mask = torch.cat([img_attention_mask, text_attention_mask])

            inputs_embeds.append(multimodal_embeds.unsqueeze(0))
            attention_mask.append(multimodal_attention_mask.unsqueeze(0))
            labels.append(multimodal_labels.unsqueeze(0))

        inputs_embeds = torch.cat(inputs_embeds)
        labels = torch.cat(labels)
        attention_mask = torch.cat(attention_mask)

        return inputs_embeds, attention_mask, labels

    def get_num_img_patches(self):
        test_input = torch.randn(1, 3, 224, 224).to(self.model_dtype)
        return self.vision_model(test_input).last_hidden_state.shape[1]
    
    def compute_split_sizes_from_input_ids(self, input_ids, image_token_id):
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        if len(image_token_indices) == 0:
            return [input_ids.size(0)]  
        split_sizes = [image_token_indices[0]]  
        split_sizes += [image_token_indices[i] - image_token_indices[i-1] for i in range(1, len(image_token_indices))]    
        split_sizes.append(input_ids.size(0) - image_token_indices[-1])
        return split_sizes

    def forward(self, pixel_values, texts):
        batch = {"pixel_values": pixel_values, "text": texts}
        inputs_embeds, attention_mask, labels = self.prepare_for_multimodal(batch)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    def generate(self, pixel_values, texts):
        batch = {"pixel_values": pixel_values, "text": texts}
        inputs_embeds, attention_mask, _ = self.prepare_for_multimodal(batch)
        generate_outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens
        )
        return self.tokenizer.batch_decode(generate_outputs, skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        loss = self(batch['pixel_values'], batch['text'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        generated_texts = self.generate(batch['pixel_values'], batch['text'])
        for i in range(len(generated_texts)):
            self.validation_results.append({
                'id': batch["id"][i],
                'generated_text': generated_texts[i],
                'original_text': batch["ground_truth"][i]
            })

    def on_validation_epoch_end(self):
        generated_texts = [result['generated_text'] for result in self.validation_results]
        original_texts = [result['original_text'] for result in self.validation_results]
        rouge_scores = self.rouge.compute(predictions=generated_texts, references=original_texts)
        rouge_l_score = rouge_scores['rougeL']
        self.log('val_rougeL', rouge_l_score)
        results_dir = f"results/validation/{self.config.dataset.replace('/', '_')}/{self.config.vision_model_name.replace('/', '_')}_{self.config.language_model_name.replace('/', '_')}"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"epoch_{self.current_epoch}_rouge_{rouge_l_score:.4f}_validation_results.json")

        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=4)

        self.validation_results.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata)).convert('RGB')
    return img 

class M3IT(Dataset):
    def __init__(self, phase, dataset, config):
        self.config = config
        self.phase = phase
        self.dataset = dataset
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.Resize((224, 224)),
            v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        pixel_values = self.transform(stringToRGB(data['image_base64_str']))
        
        if self.phase == "train":
            text = f"Human: <image> Describe this image. Assistant: {data['outputs']}"
        else:
            text = f"Human: <image> Describe this image. Assistant:"
        return {
            'id': str(uuid.uuid4()),
            'pixel_values': pixel_values,
            'text': text,
            "ground_truth": data['outputs']
        }
    
def prepare_dataset(config):
    if config.dataset == "m3it":
        dataset = load_dataset(
            "MMInstruction/M3IT", 
            "image-paragraph-captioning", 
            trust_remote_code=True
        )
        train_dataset = M3IT(
            phase='train', 
            dataset=dataset['train'].select(range(10000)),
            config=config
        )
        validation_dataset = M3IT(
            phase='validation', 
            dataset=dataset['validation'].select(range(1000)),
            config=config
        )
        test_dataset = M3IT(
            phase='test', 
            dataset=dataset['test'].select(range(1000)),
            config=config
        )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=config.test_batch_size, 
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.test_batch_size, 
        shuffle=False
    )
    return train_dataloader, validation_dataloader, test_dataloader

class Config:
    def __init__(self):
        self.tokenizer_name = "allenai/OLMo-1B-hf"
        self.language_model_name = "allenai/OLMo-1B-hf"
        self.vision_model_name = "google/siglip-base-patch16-224"
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        self.max_new_tokens = 50
        self.dataset = "m3it"
        self.max_length = 128
        self.train_batch_size = 1
        self.test_batch_size = 1
        self.num_epochs = 5
        self.wandb_run_name = self.dataset + "_" + self.vision_model_name.replace('/', '_') + "_" + self.language_model_name.replace('/', '_')

config = Config()
wandb_logger = WandbLogger(project="vlm", name=config.wandb_run_name)
model = VLM(config)
train_dataloader, validation_dataloader, test_dataloader = prepare_dataset(config)
trainer = pl.Trainer(
    max_epochs=config.num_epochs, 
    devices=1 if torch.cuda.is_available() else 0,
    logger=wandb_logger,
    log_every_n_steps=1,
    num_sanity_val_steps=10,
    accumulate_grad_batches=128,
    gradient_clip_val=1.0
)

trainer.fit(model, train_dataloader, validation_dataloader)
trainer.test(model, test_dataloader)