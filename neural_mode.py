import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Small Neural Network for regression coefficients
class NeuralCB_S(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[16, 16], activation_fn=nn.ReLU):
        super(NeuralCB_S, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
# Large Neural Network for regression coefficients
class NeuralCB_L(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[32, 32], activation_fn=nn.ReLU):
        super(NeuralCB_L, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    

class NeuralCB(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=32, model_type="Small"):
        super().__init__()
        # embeddingBag mode='mean' achieves mean pooling for multi-label features 
        self.emb_prod = nn.EmbeddingBag(vocab_sizes['prod'], embed_dim, mode='mean', padding_idx=0)
        self.emb_country = nn.EmbeddingBag(vocab_sizes['country'], embed_dim, mode='mean', padding_idx=0)
        self.emb_genre = nn.EmbeddingBag(vocab_sizes['genre'], embed_dim, mode='mean', padding_idx=0)
        self.emb_lang = nn.Embedding(vocab_sizes['lang'], embed_dim) # single label feature use simple embedding

        # compute the total input dimension for the MLP
        total_input_dim = 4 * embed_dim
        
        # call the small or large neural network based on the model type
        if model_type == "Small":
            self.mlp = NeuralCB_S(total_input_dim, 4) # output is 2,because we need Popularity and Vote
        else:
            self.mlp = NeuralCB_L(total_input_dim, 4)

    def forward(self, r, p, c, g, l):
        # compute the embeddings and mean pooling for multi-label features
        e_p = self.emb_prod(p)
        e_c = self.emb_country(c)
        e_g = self.emb_genre(g)
        e_l = self.emb_lang(l)
        
        # cat the numerical feature with all embeddings
        context = torch.cat([e_p, e_c, e_g, e_l], dim=1)
        params = self.mlp(context)

        # the regression coefficients
        w_pop  = params[:, 0:1]
        w_vote = params[:, 1:2]
        b_pop  = params[:, 2:3]
        b_vote = params[:, 3:4]
        
        # predictions
        pred_pop  = w_pop * r + b_pop
        pred_vote = w_vote * r + b_vote
        # call the MLP
        return torch.cat([pred_pop, pred_vote], dim=1), w_pop, w_vote

# define the dataset class for neural network model
class MovieDataset(Dataset):
    def __init__(self, df):
        self.r = torch.FloatTensor(df["runtime"].to_numpy().copy()).unsqueeze(1)
        self.y = torch.FloatTensor(df[["popularity", "vote_average"]].to_numpy().copy())
        self.prod = df["production_company_names_id"].tolist()
        self.country = df["origin_country_id"].tolist()
        self.genre = df["genre_names_id"].tolist()
        self.lang = torch.LongTensor(df["original_language_id"].to_numpy().copy())
        
    def __len__(self): return len(self.y)
    
    def __getitem__(self, i):
        # handle the empty multi-label features by assigning a default index 0
        prod = self.prod[i] if len(self.prod[i]) > 0 else [0]
        country = self.country[i] if len(self.country[i]) > 0 else [0]
        genre = self.genre[i] if len(self.genre[i]) > 0 else [0]
        return self.r[i], torch.LongTensor(prod), torch.LongTensor(country), torch.LongTensor(genre), self.lang[i], self.y[i]

def collate_fn(batch):
    r, p, c, g, l, y = zip(*batch)
    # Pad the sequences for multi-label features
    p_pad = pad_sequence(p, batch_first=True, padding_value=0)
    c_pad = pad_sequence(c, batch_first=True, padding_value=0)
    g_pad = pad_sequence(g, batch_first=True, padding_value=0)
    return torch.stack(r), p_pad, c_pad, g_pad, torch.stack(l), torch.stack(y)