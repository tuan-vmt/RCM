folder = "datasets/data_log"
path_file_user_info = "datasets/demographic/demographic_06.csv"
path_film_series = "datasets/data_log/tv360_film_series.csv"
path_film_episode = "datasets/data_log/tv360_film_espisode.csv"
numbers_of_hst_films = 10
numbers_of_user = 4000
prefer_threshold = 0.1
score_threshold = 0.35
device = "auto"
loss = 'normal'
epochs = 5
lr = 0.1
layer = 4
emb_size = 132
model = 'graphsage'
dict_opt = {
    'path_file_user_info': path_file_user_info,
    'folder': folder,
    'path_film_series': path_film_series,
    'path_film_episode': path_film_episode,
    'numbers_of_hst_films': numbers_of_hst_films,
    'numbers_of_user': numbers_of_user,
    'device': device,
    'prefer_threshold': prefer_threshold,
    'score_threshold': score_threshold,
    'epochs': epochs,
    'lr': lr,
    'loss': loss,
    'layer': 4,
    'emb_size': 132,
    'model': model
}
    
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


opt = DotDict(dict_opt)