folder = "demographic_data/"
path_file_user_info = "demographic_06.csv"
path_film_series = "tv360_film_series.csv"
path_film_episode = "tv360_film_episode.csv"
numbers_of_hst_films = 40
numbers_of_user = 4000
prefer_threshold = 0.1

feature_weights = [4.82780109e-02, 4.82021795e+00, 7.46684642e-01, 3.50983585e-02, 2.81441764e+02, 2.01951641e-01]
raw_features_file_dir = 'raw_features.pickle'


########hyper parameter##################
labels=None # use it with unsupervised
supervised=False
#'model: gat, graphsage (default: gat) '
model='graphsage'
weights_threshold = 0.1
n_layer=2
emb_size=128 #output of embedding size
random_state=1234
# device='auto'
device = "cuda:1"
epochs=5
batch_size=20
lr=0.7
unsup_loss_type='margin'
print_progress=True



dict_opt = {
    'path_file_user_info': path_file_user_info,
    'folder': folder,
    'path_film_series': path_film_series,
    'path_film_episode': path_film_episode,
    'numbers_of_hst_films': numbers_of_hst_films,
    'numbers_of_user': numbers_of_user,
    'device': device,
    'prefer_threshold': prefer_threshold
}
    
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


opt = DotDict(dict_opt)