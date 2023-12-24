import os
import json
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import CLIPProcessor, CLIPModel



# Create a dataset split 
def create_dataset_split(json_path, image_path, preprocess, split_type, batch_size=1000, shuffle=True):
    # Read json data
    with open(json_path, 'r') as f:
        input_data = []
        for line in f:
            obj = json.loads(line)
            input_data.append(obj)


    # Define a custom dataset
    class CustomDataset():
        def __init__(self, list_image_path, list_txt, split_type):
            # Initialize image paths and corresponding texts
            self.image_path = list_image_path
            if split_type == 'train' or split_type == 'val':
                # Tokenize text using CLIP's tokenizer
                self.title  = clip.tokenize(list_txt)
            else:
                self.title = list_txt # This for the testing!

        def __len__(self):
            return len(self.title)

        def __getitem__(self, idx):
            # Preprocess image using CLIP's preprocessing function
            image = preprocess(Image.open(self.image_path[idx]))
            title = self.title[idx]
            return image, title
        

    # create split 
    list_image_path = []
    list_txt = []
    for item in input_data:
        img_path = os.path.join(image_path, item['image_path'].split('/')[-1])
        caption = item['product_title'][:40]
        list_image_path.append(img_path)
        list_txt.append(caption)


    dataset = CustomDataset(list_image_path, list_txt, split_type=split_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 

    return dataloader
    


# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 



@torch.no_grad()
def test(test_loader, comparison_list, device):#TODO: Complete the similarity results!

    # Loading the saved model
    chkpt_path = '/home/vimukthi/Myprojects/Experiments/CLIP-SAM/model_checkpoints/finetuned_clip_model.pth' 

    model, _ = clip.load(chkpt_path, device, jit=False)

    pbar = tqdm(test_loader, total=len(test_loader))

    for batch in pbar:
        images, texts = batch 
            
        images = images.to(device)
        texts  = torch.cat([clip.tokenize(f"a photo of a {c}") for c in comparison_list]).to(device)

        # Encode image and text
        image_feature_arr = model.encode_image(images)
        text_feature_arr = model.encode_text(texts)

        image_feature_arr /= image_feature_arr.norm(dim=-1, keepdim=True)
        text_feature_arr /= text_feature_arr.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feature_arr @ text_feature_arr.T).softmax(dim=-1)
        print(similarity)
        # values, indices = similarity[0].topk(5)

        # # Print the top predictions
        # print("\nTop predictions:\n")
        # for value, index in zip(values, indices):
        #     print(f"{comparison_list[index]:>16s}: {100 * value.item():.2f}%")



def train(train_dataloader, model, device, num_epochs=30):

    if device == "cpu":
        model.float()

    # Prepare the optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=5e-5,
                                betas=(0.9,0.98),
                                eps=1e-6,
                                weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset


    # Specify the loss function
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    model.train()
    # Train the model
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in pbar:
            optimizer.zero_grad()

            images, texts = batch 
            
            images = images.to(device)
            texts  = texts.to(device)

            # Forward pass
            logits_per_image, logits_per_text = model(images, texts)

            # Compute loss
            ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            # Backward pass
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

    os.makedirs('./model_checkpoints', exist_ok=True)
    torch.save(model.state_dict(), './model_checkpoints/finetuned_clip_model.pth')




if __name__ == '__main__':
    # Load the CLIP model and processor
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    # Choose computation device
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 

    # Load pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_split_json_path = '/home/vimukthi/Myprojects/Experiments/CLIP-SAM/data/sample_dataset/train_data.json'
    val_split_json_path = '/home/vimukthi/Myprojects/Experiments/CLIP-SAM/data/sample_dataset/val_data.json'
    test_split_json_path = '/home/vimukthi/Myprojects/Experiments/CLIP-SAM/data/sample_dataset/test_data.json'

    train_images_path = '/home/vimukthi/Myprojects/Experiments/CLIP-SAM/data/sample_dataset/images/train'
    val_images_path = '/home/vimukthi/Myprojects/Experiments/CLIP-SAM/data/sample_dataset/images/val'
    test_images_path = '/home/vimukthi/Myprojects/Experiments/CLIP-SAM/data/sample_dataset/images/test'

    train_dataloader = create_dataset_split(json_path=train_split_json_path, 
                                            image_path=train_images_path, 
                                            preprocess=preprocess, 
                                            split_type='train')

    val_dataloader = create_dataset_split(json_path=val_split_json_path, 
                                          image_path=val_images_path, 
                                          preprocess=preprocess, 
                                          split_type='val')

    test_dataloader = create_dataset_split(json_path=test_split_json_path, 
                                           image_path=test_images_path, 
                                           preprocess=preprocess, 
                                           split_type='test')

    train(train_dataloader=train_dataloader, model=model, device=device, num_epochs=30)
    
    comparison_list = [
        'Saree',
        'Trouser',
        'Gown'
    ]
    test(test_loader=test_dataloader, comparison_list=comparison_list, device=device)