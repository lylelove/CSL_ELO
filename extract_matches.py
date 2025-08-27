import json

def extract_match_data():
    # 读取原始数据文件
    with open('matches.json', 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    # 提取需要的字段
    extracted_data = []
    for match in matches:
        extracted_match = {
            'team_A_name': match.get('team_A_name'),
            'team_B_name': match.get('team_B_name'),
            'score_A': match.get('score_A'),
            'score_B': match.get('score_B'),
            'minute_extra': match.get('minute_extra')
        }
        extracted_data.append(extracted_match)
    
    # 保存到新的JSON文件
    with open('extracted_matches.json', 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print(f"提取完成，共处理 {len(extracted_data)} 条数据，已保存到 extracted_matches.json")

if __name__ == "__main__":
    extract_match_data()