import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer import Tokenizer
from dataset import TranslationDataset, collate_fn
from model import Transformer
from config import config

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=True)
    for batch_idx, (src, decoder_input, decoder_target) in enumerate(loop):
        src           = src.to(device)
        decoder_input = decoder_input.to(device)
        decoder_target = decoder_target.to(device)
        output = model(src, decoder_input)
        output         = output.view(-1, output.shape[-1])
        decoder_target = decoder_target.view(-1)
        loss = criterion(output, decoder_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    loop = tqdm(dataloader, desc="Evaluating", leave=True)
    with torch.no_grad():
        for src, decoder_input, decoder_target in loop:
            src           = src.to(device)
            decoder_input = decoder_input.to(device)
            decoder_target = decoder_target.to(device)
            output = model(src, decoder_input)
            output         = output.view(-1, output.shape[-1])
            decoder_target = decoder_target.view(-1)
            loss = criterion(output, decoder_target)
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"using: {device}")

    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()

    train_dataset = TranslationDataset("train", src_tokenizer, tgt_tokenizer)
    val_dataset   = TranslationDataset("validation", src_tokenizer, tgt_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], 
                              shuffle=False, collate_fn=collate_fn)

    model = Transformer(
        src_vocab=len(src_tokenizer.word2idx),
        tgt_vocab=len(tgt_tokenizer.word2idx),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    total = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total:,}")

    for epoch in range(config["num_epochs"]):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch+1}/{config['num_epochs']} → train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")
        
        # save checkpoint every epoch
        import os
        os.makedirs("/kaggle/working", exist_ok=True)
        torch.save(model.state_dict(), f"/kaggle/working/model_epoch_{epoch+1}.pt")
        print(f"model saved → model_epoch_{epoch+1}.pt")