import json

squad_json_path = 'data/train-v1.1.json'  # Or dev-v1.1.json if you wish
output_txt_path = 'data/squad_contexts.txt'

with open(squad_json_path, 'r', encoding='utf-8') as file:
    squad_data = json.load(file)

contexts = []
for article in squad_data['data']:
    for paragraph in article['paragraphs']:
        context = paragraph['context'].replace('\n', ' ').strip()
        contexts.append(context)

with open(output_txt_path, 'w', encoding='utf-8') as out_file:
    for context in contexts:
        out_file.write(context + '\n\n')

print(f"Extracted {len(contexts)} context paragraphs. Saved to {output_txt_path}")


