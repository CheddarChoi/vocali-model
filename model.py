import csv, os, random, pickle, keras
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import List, Optional

moods = ['happy', 'energetic', 'depression', 'calm']
int2pitch = {0: ["C"],1: ["C#"],2: ["D"],3: ["D#"],4: ["E"],5: ["F"],
             6: ["F#"],7: ["G"],8: ["G#"],9: ["A"],10: ["A#"],11: ["B"]}
pitch2int = {"C": 0,"B#": 0,"C#":1,"D": 2,"D#": 3,"E": 4,"F": 5,
             "E#": 5,"F#": 6,"G": 7,"G#": 8,"A": 9,"A#": 10,"B": 11}

context = {}

# Gets the list of all track ids and names
def get_final_track_list(filename):
    with open(filename, 'r', encoding='utf-8') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      final_tracks_ids = []
      final_tracks_names = []
      line_count = 0
      
      for row in csv_reader:
        if (line_count > 0):
          final_tracks_ids.append(row[6])
          final_tracks_names.append(row[1])
        line_count += 1

    return final_tracks_ids, final_tracks_names

# Loads the dataset of user-item data
def load_data(filename):
  with open(filename, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    items = []
    users = []
    final_tracks_ids, final_tracks_names = get_final_track_list("./songListWithFeatures.csv")
    tracks = []
    ratings = []

    for row in csv_reader:
      if (line_count == 2):
        users = row[3:-2]
      if (line_count > 2):
        if (row[-1] in final_tracks_ids):
          tracks.append(row[-1])
          ratings.append(row[3:-2])
      line_count += 1

    for track_id in range(len(tracks)):
      for user_id in range(len(ratings[track_id])):
        if (ratings[track_id][user_id] == 'TRUE'):
          item = [user_id, track_id, 1]
        else:
          item = [user_id, track_id, 0.5]
        items.append(item)
    
    return users, tracks, items

# Getting the user-item sparse matrix
def get_user_item_sparse_matrix(data):
    sparse_data = sparse.csr_matrix((data.rating, (data.user, data.track)))
    return sparse_data

# Calculatest the average rating of a user
def get_average_rating(sparse_matrix, is_user):
    ax = 1 if is_user else 0
    sum_of_ratings = sparse_matrix.sum(axis = ax).A1  
    no_of_ratings = sparse_matrix.sum(axis = ax).A1 
    rows, cols = sparse_matrix.shape
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i] for i in range(rows if is_user else cols) if no_of_ratings[i] != 0}
    return average_ratings

# Calculates the similar features of top 10 user and tracks
def create_new_similar_features_for_new_user(sample_sparse_matrix, new_user):
    global_avg_rating = get_average_rating(sample_sparse_matrix, False)
    global_avg_users = get_average_rating(sample_sparse_matrix, True)
    global_avg_tracks = get_average_rating(sample_sparse_matrix, False)
    sample_train_users, sample_train_tracks, sample_train_ratings = sparse.find(sample_sparse_matrix)
    new_features_csv_file = open("../new_features_for_user.csv", mode = "w")
    
    for user, track, rating in zip(sample_train_users, sample_train_tracks, sample_train_ratings):
      if (user == new_user):
        similar_arr = list()
        similar_arr.append(user)
        similar_arr.append(track)
        #similar_arr.append(sample_sparse_matrix.sum()/sample_sparse_matrix.count_nonzero())
        
        similar_users = cosine_similarity(sample_sparse_matrix[user], sample_sparse_matrix).ravel()
        indices = np.argsort(-similar_users)[1:]
        ratings = sample_sparse_matrix[indices, track].toarray().ravel()
        top_similar_user_ratings = list(ratings[:5])
        top_similar_user_ratings.extend([global_avg_rating[track]] * (5-len(ratings)))
        similar_arr.extend(top_similar_user_ratings)
        
        similar_tracks = cosine_similarity(sample_sparse_matrix[:,track].T, sample_sparse_matrix.T).ravel()
        similar_tracks_indices = np.argsort(-similar_tracks)[1:]
        similar_tracks_ratings = sample_sparse_matrix[user, similar_tracks_indices].toarray().ravel()
        top_similar_track_ratings = list(similar_tracks_ratings[:5])
        top_similar_track_ratings.extend([global_avg_users[user]] * (5-len(top_similar_track_ratings)))
        similar_arr.extend(top_similar_track_ratings)
        
        #similar_arr.append(global_avg_users[user])
        #similar_arr.append(global_avg_tracks[track])
        similar_arr.append(rating)
        
        new_features_csv_file.write(",".join(map(str, similar_arr)))
        new_features_csv_file.write("\n")
        
    new_features_csv_file.close()
    new_features_df = pd.read_csv('../new_features_for_user.csv', names = ["user_id", "track_id", "similar_user_rating1", 
                                                               "similar_user_rating2", "similar_user_rating3", "similar_user_rating4", "similar_user_rating5",
                                                               "similar_track_rating1", "similar_track_rating2", 
                                                               "similar_track_rating3", "similar_track_rating4", "similar_track_rating5",
                                                               "rating"])
    
    return new_features_df


def init_model():
    print("Init model start")
    final_tracks_ids, final_tracks_names = get_final_track_list("./songListWithFeatures.csv")
    users, tracks, items = load_data('./userData.csv')
    data = pd.DataFrame(items, columns=["user", "track", "rating"])
    context['data'] = data
    context['final_tracks_ids'] = final_tracks_ids

    with open("clf.pkl", 'rb') as file:
        clf = pickle.load(file)
    context['clf'] = clf

    trained_features = pd.read_csv('trained_features.csv')
    context['trained_features'] = trained_features

    # Mood Analysis
    mood_lrs = {}
    for mood in moods:
        pkl_filename = "LR_"+mood+".pkl"
        with open(pkl_filename, 'rb') as file:
            mood_lrs[mood] = pickle.load(file)
    context['mood_lrs'] = mood_lrs

    df_songs = pd.read_csv('songListWithFeatures.csv',index_col=['num'])
    song_num = len(df_songs)
    context['song_num'] = song_num
    context['df_songs'] = df_songs

    print("Init model end")

def send_output(newWeight, liked, disliked, undefined, minPitch, maxPitch, newMood):
  song_num = context['song_num']
  df_songs = context['df_songs']
  trained_features = context['trained_features']
  clf = context['clf']
  mood_lrs = context['mood_lrs']
  data = context['data']
  final_tracks_ids = context['final_tracks_ids']

  # Get input
  weight = newWeight
  user_likeList = liked
  user_dislikeList = disliked
  user_undefinedList = undefined
  user_minPitch = minPitch
  user_maxPitch = maxPitch
  user_mood = newMood[0]
  user_with = newMood[1]

  ## Mood
  # Initializing scores
  mood_score = [0] * song_num

  # Mood analysis for each songs with LR
  features = ['loudness','mode','speechiness','acousticness','instrumentalness',
              'liveness','valence','tempo']

  for i in range(song_num):
    row = df_songs.iloc[[i]]
    # print(str(i)+" : "+row['title'])
    song_features = row[features]
    score = 0
    if user_with == "alone":
      score += mood_lrs[user_mood].predict_proba(song_features)[0][1]
    elif user_with == "together":
      score += mood_lrs[user_mood].predict_proba(song_features)[0][1] * 0.6
      score += mood_lrs["energetic"].predict_proba(song_features)[0][1] * 0.2
      score += mood_lrs["happy"].predict_proba(song_features)[0][1] * 0.2
    mood_score[i] += score

  ## Song Preference
  # Get predictions for the new user for collective filtering
  new_items = []
  for i in range(len(final_tracks_ids)):
    if (final_tracks_ids[i] in user_likeList):
      new_items.append([50, i, 1])
    elif (final_tracks_ids[i] in user_dislikeList):
      new_items.append([50, i, -1])
    else:
      new_items.append([50, i, 0.5])

  new_data = pd.DataFrame(new_items, columns=["user", "track", "rating"])

  new_user_sparse_matrix = get_user_item_sparse_matrix(pd.concat([new_data, data]))
  new_features = create_new_similar_features_for_new_user(new_user_sparse_matrix, 50)
  new_final_features = pd.concat([new_features, trained_features])
  new_test = new_final_features.loc[new_final_features['user_id']==50].drop(["user_id", "track_id", "rating"], axis = 1)
  new_pred_test = clf.predict(new_test)
  preference_score = new_pred_test

  # Pitch Analysis
  pitch_score = [0] * song_num
  # Calculate user key
  user_minKey = 11 * int(user_minPitch[-1]) + pitch2int[user_minPitch[:-1]]
  user_maxKey = 11 * int(user_maxPitch[-1]) + pitch2int[user_maxPitch[:-1]]
  user_key = int((user_minKey + user_maxKey)/2) % 11
  for i in range(song_num):
    song_key = int(df_songs.iloc[i]['key'])
    pitch_score[i] = (12 - min(abs(user_key-song_key), 12 - abs(user_key-song_key))) / 24

  ## Score Calculation
  total_score = [0] * song_num
  for i in range(song_num):
    total_score[i] = weight[0] * preference_score[i] + weight[1] * mood_score[i] + weight[2] * pitch_score[i]
    # Make already rated songs -1
    id = df_songs.iloc[i]['id']
    if (id in user_likeList) or (id in user_dislikeList) or (id in user_undefinedList):
      total_score[i] = -1

  # Print Top 10 results
  df_scores = pd.DataFrame(total_score, columns = ['score'])
  top_scores = list(df_scores.sort_values('score', ascending = False).index)
  rec_list = df_songs.iloc[top_scores]
  rec_list = rec_list[~rec_list['id'].isin(user_likeList)]
  rec_list = rec_list[['title', 'artist', 'id']]

  top10_mood_score = []
  top10_preference_score = []
  top10_pitch_score = []
  top10_total_score = []

  top10_indices = list(rec_list.index.values)

  for idx in top10_indices:
    top10_mood_score.append(mood_score[idx])
    top10_preference_score.append(preference_score[idx])
    top10_pitch_score.append(pitch_score[idx])
    top10_total_score.append(total_score[idx])

  rec_list['mood_score'] = top10_mood_score
  rec_list['preference_score'] = top10_preference_score
  rec_list['pitch_score'] = top10_pitch_score
  rec_list['total_score'] = top10_total_score

  return rec_list