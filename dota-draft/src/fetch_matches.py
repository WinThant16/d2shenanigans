import requests
import pandas as pd
import time
import os

url = "https://api.opendota.com/api/publicMatches"

payload = {'less_than_match_id':'8891729002', 'min_rank':70, 'max_rank':80}

def is_this_match_valid(match):
	# check valid game duration
	duration = match['duration']

	# check lobby type is all pick (normal or ranked)
	# id for normal: 0, id for ranked: 7
	lobby_type = match['lobby_type']
	# check valid game mode: all pick: 1, captains mode: 2, random draft: 3, single draft: 4, all random: 5, all draft: 22.
	game_mode = match['game_mode']

	# check valid heroes for radiant team
	radiant_valid = all(hero != 0 for hero in match['radiant_team']) and len(match['radiant_team']) == 5
	# check valid heroes for dire team
	dire_valid = all(hero != 0 for hero in match['dire_team']) and len(match['dire_team']) == 5

	if (duration > 0) and (lobby_type == 7) and (game_mode == 22) and radiant_valid and dire_valid:
		return True
	
	return False


file_path = "data/raw/valid-matches.parquet"

if os.path.exists(file_path):
	df = pd.read_parquet(file_path)
	all_matches = df.to_dict('records')
	starting_cursor = df['match_id'].min()
else:
	all_matches = []
	starting_cursor = 8891729002

payload['less_than_match_id'] = starting_cursor
retry_counter = 0
while(len(all_matches) < 50000):
	try:
		# get request to fetch data
		response = requests.get(url, timeout=30, params=payload)
		# error if http request failed (404/500)
		response.raise_for_status()
		# extract and use response data
		data = response.json()

	except requests.exceptions.RequestException as error:
		print(f"Error retrieving data {error}")
		if (retry_counter >= 5):
			break
		print("Waiting 5 seconds before moving to the next step...")
		time.sleep(5) 
		retry_counter +=1
		print("Resuming script execution...")
		continue
				
	# validate matches as ranked
	retry_counter = 0

	valid_matches = [match for match in data if is_this_match_valid(match)]
		
	all_matches.extend(valid_matches)

	# update last match id to recurse
	cursor = min(match['match_id'] for match in data)
	payload['less_than_match_id'] = cursor

	df = pd.DataFrame(all_matches).drop_duplicates(subset="match_id")
	all_matches = df.to_dict('records')
	df.to_parquet(file_path)
	time.sleep(1.1)
	print(f"{len(all_matches)} collected so far.")

if (retry_counter >= 5):
	print("Encountered error while retrieving.")
else:
	print("Data successfully retrieved.")