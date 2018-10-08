## 技術書典5
技術書典5で頒布する`実装しながら深層学習でGANの画像変換技術を学ぼう`のソースコードです.
U-Netとpix2pixの実装がしてあります.
データセットはダウンロードカードよりダウンロードしてください.

### 準備
Anaconda環境の整備
1. `pyenv install --list | anaconda`で好きな環境を入れてください.
2. CUDA8.0の場合
`conda install pytorch torchvision cuda80 -c pytorch`
3. CUDA9.0の場合
`conda install pytorch torchvision -c pytorch`

pyenvなどはこちらを参考にしてください.[https://qiita.com/udoooom/items/fa3c7554556831b2d65a](https://qiita.com/udoooom/items/fa3c7554556831b2d65a)