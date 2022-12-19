folder = "/home/gemai/mnt_raid1/tuanvm/Viettel/demographic_data/"
path_file_user_info = "demographic_06.csv"
path_film_series = "tv360_film_series.csv"
path_film_episode = "tv360_film_episode.csv"
numbers_of_hst_films = 10
numbers_of_user = 4000
duration_threshold = 300
threshold_category = 4
threshold_actor = 4
threshold_director = 1
threshold_nb_of_user = 10
device = "cuda:0"

dict_opt = {
    'path_file_user_info': path_file_user_info,
    'folder': folder,
    'path_film_series': path_film_series,
    'path_film_episode': path_film_episode,
    'numbers_of_hst_films': numbers_of_hst_films,
    'numbers_of_user': numbers_of_user,
    'device': device,
    'duration_threshold': duration_threshold,
    'threshold_category': threshold_category,
    'threshold_actor': threshold_actor,
    'threshold_director': threshold_director,
    "threshold_nb_of_user": threshold_nb_of_user
}
    
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


opt = DotDict(dict_opt)