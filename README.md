# Thunderstorm Sound

## 雨の降る音、雷が遠くで鳴る音をPythonで再現しました
**WAVファイルです**Pythonを実行する度に音がランダムに再生されます
ただし、1度ファイルを保存したらその音で変わらず聴くことになります

必要なライブラリです。Bash↓↓↓
```
python3 -m pip install numpy scipy
```
実行↓↓↓
```
python3 random_light_rain_thunder.py
```
長さ変える時↓↓↓
```
python3 random_light_rain_thunder.py --duration 120
```
ファイル名固定するとき↓↓↓
```
python3 random_light_rain_thunder.py \
  --output rain.wav
```

### シード値固定してません


~~README、ChatGPTに書かせれば良かった~~

## 英語版欲しかったらIssueください
## If you'd like an English version, please open an issue.

あああ
