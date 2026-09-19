
import pandas as pd
import jiwer
df = pd.read_csv('results/tables/whisper_results.csv')
df['source'] = df['audio_path'].apply(lambda p: 'neural' if 'neural' in p else 'sapi5')
df['lang'] = df['audio_path'].apply(lambda p: p.split('_')[-2])

results = []
for source in ['sapi5', 'neural']:
    for lang in ['en', 'fr', 'ar']:
        sub = df[(df['source'] == source) & (df['lang'] == lang)]
        if len(sub) == 0: continue
        wer = jiwer.wer(sub['reference'].tolist(), sub['hypothesis'].tolist())
        cer = jiwer.cer(sub['reference'].tolist(), sub['hypothesis'].tolist())
        results.append({'source': source, 'lang': lang, 'n': len(sub), 'wer': round(wer, 4), 'cer': round(cer, 4)})
print(pd.DataFrame(results).to_string(index=False))
