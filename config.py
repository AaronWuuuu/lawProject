import torch 

class config:
    batch_size = 16
    lr = 1e-3
    model_path = 'E:/learning_material/top_xhs/bertClassForLawMultiTask/base_model_file'
    data_path = 'E:/learning_material/top_xhs/bertClassForLawMultiTask/data/data_final'
    epochs = 3
    max_len = 512
    input_size = 768
    hidden_size = 256
    accusation_num = 116
    relevant_articles_num = 100
    
    #Todo : deepspeed applied
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
