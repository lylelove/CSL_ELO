import requests
import json

def fetch_matches():
    grouped_matches = {}
    
    # 循环获取 1 到 30 轮的数据
    for gameweek in range(1, 31):
        url = f"https://sport-data.dongqiudi.com/soccer/biz/data/schedule?season_id=23540&round_id=348912&gameweek={gameweek}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if 'content' in data and 'matches' in data['content']:
                matches = data['content']['matches']
                
                simplified_matches = []
                for match in matches:
                    simplified_match = {
                        'team_A_name': match.get('team_A_name', ''),
                        'team_B_name': match.get('team_B_name', ''),
                        'score_A': match.get('score_A', ''),
                        'score_B': match.get('score_B', '')
                    }
                    simplified_matches.append(simplified_match)
                
                grouped_matches[f'gameweek_{gameweek}'] = simplified_matches
                print(f"成功获取第 {gameweek} 轮数据，共 {len(simplified_matches)} 场比赛")
            else:
                print(f"第 {gameweek} 轮数据格式异常")
                
        except requests.exceptions.RequestException as e:
            print(f"请求第 {gameweek} 轮数据时出错: {e}")
        except json.JSONDecodeError as e:
            print(f"解析第 {gameweek} 轮JSON数据时出错: {e}")
    
    with open('matches.json', 'w', encoding='utf-8') as f:
        json.dump(grouped_matches, f, ensure_ascii=False, indent=2)
    
    total_matches = sum(len(matches) for matches in grouped_matches.values())
    print(f"数据获取完成，总共 {total_matches} 场比赛，已按轮次分组保存到 matches.json")

if __name__ == "__main__":
    fetch_matches()