from transformers import AutoModel
import torch.nn as nn
import torch


class lawModel(nn.Module):
    def __init__(self, config):
        super(lawModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained(config.model_path)
        
        # freeze bert grad
        for p in self.bert_model.parameters():
            p.required_grad = False
        
        self.fc_accusation = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.accusation_num),
            
        )
        
        self.fc_relevant_articles = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.relevant_articles_num),
            
        )
                
        self.fc_imprisonment = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )
        
        self.public_fc = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.input_size)
        )
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        x = self.bert_model(input_ids, token_type_ids, attention_mask)['pooler_output']
        x = self.public_fc(x)
        
        accusation_out = self.fc_accusation(x)
        relevant_articles_out = self.fc_relevant_articles(x)
        imprisonment_out = self.fc_imprisonment(x)
        imprisonment_out = torch.squeeze(imprisonment_out, dim=1) 
        return accusation_out, relevant_articles_out, imprisonment_out
        
        
if __name__ == '__main__':
    from config import config
    config1 = config()
    model = lawModel(config1)
    
    from config import config
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, AutoConfig


    config1 = config()

    Tokenizer = AutoTokenizer.from_pretrained(config1.model_path)
    from dataProcess import dataSetTorch, dataPreProcess

    data_loader = dataPreProcess(config1)
    data = data_loader.getOriData()
    print(data['relevant_articles'][1])

    dataSet = dataSetTorch(Tokenizer, data, config1)

    dataloader_se = DataLoader(dataSet, batch_size=config1.batch_size, shuffle=True)
    input_ids, token_type_ids, attention_mask, accusation_label, relevant_articles_label, imprisonment_label = next(iter(dataloader_se))
    
    
    print(model(input_ids, token_type_ids, attention_mask)[2].shape)
        

