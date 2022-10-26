folder = "demographic_data/"
path_file_user_info = "demographic_06.csv"
path_film_series = "tv360_film_series.csv"
path_film_episode = "tv360_film_episode.csv"
numbers_of_hst_films = 5
numbers_of_user = 4000
device = "cuda:0"

dict_opt = {
    'path_file_user_info': path_file_user_info,
    'folder': folder,
    'path_film_series': path_film_series,
    'path_film_episode': path_film_episode,
    'numbers_of_hst_films': numbers_of_hst_films,
    'numbers_of_user': numbers_of_user,
    'device': device
}
    
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


opt = DotDict(dict_opt)