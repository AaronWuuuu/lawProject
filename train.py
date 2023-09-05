
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

# from sklearn.model_selection import train_test_split

from transformers import  AutoTokenizer
from config import config

from torch.optim import Adam

from dataProcess import dataPreProcess, dataSetTorch
from model import lawModel

from torch.cuda.amp import GradScaler, autocast

from torch.utils.tensorboard import SummaryWriter

def train():
    #Todo:add argument configuration
    # config    
    train_config = config()
    
    # init tensorboard
    writer = SummaryWriter(config.log_path)
    
    # init Tokenizer
    Tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
    
    #Todo:rebuild the data load method, make it can distinguish trainSet and vaildSet.
    # load data
    data_loader = dataPreProcess(train_config)
    data = data_loader.getOriData()
    dataSet = dataSetTorch(Tokenizer, data, train_config)
    trainLoader = DataLoader(dataSet, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # init model
    model = lawModel(train_config).to(device=config.device)
    
    # init optim、creterion_cro_entro、creterion_reg
    optimizer = Adam(model.parameters(), lr=config.batch_size)
    creterion_cro_entro = nn.CrossEntropyLoss()
    creterion_reg = nn.MSELoss()
    
    GradScale = GradScaler()
    
    step = 0
    for epoch in range(config.epochs):
        with tqdm(total=len(trainLoader)) as p_bar:
            for batch_idx, (input_ids, token_type_ids, attention_mask, accusation_label, relevant_articles_label, imprisonment_label) in enumerate(trainLoader):
                input_ids = input_ids.to(device=config.device)
                token_type_ids = token_type_ids.to(device=config.device)
                attention_mask = attention_mask.to(device=config.device)
                accusation_label = accusation_label.to(device=config.device)
                relevant_articles_label = relevant_articles_label.to(device=config.device)
                imprisonment_label = imprisonment_label.to(device=config.device)
                
                with autocast():
                    accusation_out, relevant_articles_out, imprisonment_out = model(input_ids, token_type_ids, attention_mask)
                    loss_accusation = creterion_cro_entro(accusation_out, accusation_label.long())
                    loss_relevant_articles = creterion_cro_entro(relevant_articles_out, relevant_articles_label.long())
                    # loss_imprisonment_label = creterion_reg(imprisonment_out, imprisonment_label)
                    
                    #Todo : cascade structure needed, the weights of loss need to be determined.
                    loss_total = loss_accusation + loss_relevant_articles
                
                GradScale.scale(loss_total).backward()
                GradScale.step(optimizer) 
                GradScale.update()
                
                writer.add_scalar('accusation loss', loss_accusation, global_step=step)
                writer.add_scalar('relevant_articles loss', loss_relevant_articles, global_step=step)
                # writer.add_scalar('imprisonment loss', loss_imprisonment_label, global_step=step)
                # writer.add_scalar('total loss', loss_total.detach().cpu().numpy(), global_step=step)
                
                step += 1
                p_bar.set_description(f"step : {step}")
                p_bar.set_postfix(
                    loss_accusation=loss_accusation.detach().cpu().numpy(),
                    loss_relevant_articles=loss_relevant_articles.detach().cpu().numpy(), 
                    # loss_imprisonment_label=loss_imprisonment_label.detach().cpu().numpy(), 
                    )
                p_bar.update()

if __name__ == '__main__':
    train()