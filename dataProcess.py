from datasets import load_dataset
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch

from transformers import AutoModel, AutoTokenizer, AutoConfig

class dataPreProcess(object):
    def __init__(self, train_config):
        self.train_config = train_config
        self.rawdata = load_dataset(
            "json",
            data_dir= train_config.data_path
        )
        self.accusation_label_encoder = preprocessing.LabelEncoder()
        self.relevant_articles_label_encoder = preprocessing.LabelEncoder()
        
    def getOriData(self):
        relevant_articles = []
        accusation = []
        imprisonment = []
        fact = self.rawdata['train']['fact']
        meta = self.rawdata['train']['meta']
        

        for i in tqdm(range(len(meta))):
            accusation.append(meta[i]['accusation'])
            relevant_articles.append(meta[i]['relevant_articles'])
            imprisonment.append((meta[i]['term_of_imprisonment']['imprisonment']))
        
        accusation_label = self.accusation_label_encoder.fit_transform(accusation)
        relevant_articles_label = self.accusation_label_encoder.fit_transform(relevant_articles)
        
        return {
            'fact':fact,
            'accusation':accusation_label,
            'relevant_articles':relevant_articles_label,
            'imprisonment':imprisonment,
                }
        
    def __len___(self):
        return len(self.fact)
    
class dataSetTorch(Dataset):
    def __init__(self, Tokenizer, data, train_config) -> None:
        self.Tokenizer = Tokenizer
        self.data = data
        self.train_config = train_config
        
    def __len__(self):
        return len(self.data['fact'])
    
    def __getitem__(self, index):
        fact = self.data['fact'][index]
        fact_token = self.Tokenizer.encode_plus(
            fact,
            max_length=self.train_config.max_len, 
            truncation=True, 
            add_special_tokens=True,
            padding='max_length'
        )
        
        input_ids = torch.tensor(fact_token['input_ids'])
        token_type_ids = torch.tensor(fact_token['token_type_ids'])
        attention_mask = torch.tensor(fact_token['attention_mask'])
        
        accusation_label = torch.tensor(self.data['accusation'][index])
        relevant_articles_label = torch.tensor(self.data['relevant_articles'][index])
        imprisonment_label = torch.tensor(self.data['imprisonment'][index], dtype=torch.float32)
        return input_ids, token_type_ids, attention_mask, accusation_label, relevant_articles_label, imprisonment_label
        
        
        
if __name__  == '__main__':
    from config import config
    train_config = config()
    
    Tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
    
    data_loader = dataPreProcess(train_config)
    data = data_loader.getOriData()
    print(data['relevant_articles'][2])
    
    dataSet = dataSetTorch(Tokenizer, data, train_config)
    
    dataloader_se = DataLoader(dataSet, batch_size=train_config.batch_size, shuffle=True)
    
    print(next(iter(dataloader_se)))
    