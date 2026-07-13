import requests

url = "https://api.opendota.com/api/publicMatches"

payload = {'less_than_match_id':'8892169836', 'min_rank':70, 'max_rank':80}

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

# poison = {'match_id': 8892169771, 'duration': 0, 'lobby_type': 4, 'game_mode': 1,
#           'radiant_team': [0,0,0,0,0], 'dire_team': [0,0,0,0,0]}
# print(is_this_match_valid(poison))   # expect False

# good = {'match_id': 8892169790, 'duration': 2454, 'lobby_type': 7, 'game_mode': 22,
#         'radiant_team': [63,98,111,106,74], 'dire_team': [35,5,25,39,128]}
# print(is_this_match_valid(good))     # expect True


try: 
	# get request to fetch data
	response = requests.get(url, timeout=30, params=payload)
	# error if http request failed (404/500)
	response.raise_for_status()
	# extract and use response data
	data = response.json()
	valid_matches = [match for match in data if is_this_match_valid(match)]
	print("Data successfully retrieved.")
	print(f"Retrieved {len(data)} matches, {len(valid_matches)} valid after filtering.")


except requests.exceptions.RequestException as error:
	print(f"Error retrieving data {error}")