# II. seminární práce z předmětu Počítačové zpracování signálu (KI/PZS)

- Tým: Martin Formánek, Radek Šmejkal
- 28.01.2025
- [Úloha 1](sem.ipynb)
- [Zadání](SeminarniPrace-II.pdf)

## Zadání - Detekce a přiřazení slov v záznamu řeči

Ve zdrojovém souboru _Signal1.txt_ a na Obrázku 1 najdete záznam řeči převedený na signál,
resp. časovou řadu. Pomocí metod analýzy signálu v časové oblasti, frekvenční oblasti nebo
jejich kombinací identifikujte jednotlivá slova v záznamu. Vybírejte z následujícího seznamu
slov:

```text
time, prepare, solution, make, mistake, no, the, probable, long, lecture, method, disaster, fail,
work, advice, idea, succeed, easy, is, for, give
```

Vámi navržený algoritmus vyzkoušejte na záznamech _Signal2.txt_ a _Signal3.txt_ a identifikujte
slova i v těchto dvou časových řadách. V případě potřeby algoritmus dále vylepšete. Kromě
metod probraných při hodinách lze pro identifikaci jednotlivých slov využít například některé
další funkce, například Hammingova funkce nebo Hilbertova transformace, případně jakoukoli
další metodu, kterou uznáte za vhodnou, vyjma metod založených na strojovém učení.

Vzorkovací frekvence signálu je ve všech případech 22050 Hz.

### Řešení

Jako první jsem si vygeneroval `.wav` soubory jednotlivých slov a poté jsem si je uložil jako signál `scipy.io.wavfile.read(signal_file)`.  
Nejprve ze signálu odstraním prázdné části `trim_speech(audio_data[current_pos : (idx + 1) * segment_size])` a následně ho rozdělím na jednotlivá slova `segment_audio(data, segment_size, amplitude_threshold)`. Následně z každého segmentu extrahuji jeho audio charakteristiky:

```py
sY = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft)
mfccsY = librosa.feature.mfcc(S=librosa.power_to_db(sY), n_mfcc=n_mfcc)
scY = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
chromaY = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)
```

Nakonec porovnám tyto charakteristiky pomocí directed Hauserdorff distance.

```py
dist_mfcc = directed_hausdorff(features_ref[0], features_seg[0])[0]
dist_sc = directed_hausdorff(features_ref[1], features_seg[1])[0]
dist_chroma = directed_hausdorff(features_ref[2], features_seg[2])[0]
```

Na základě těchto výsledků sestavuji seznam potenciálních slov v databázi a ze seznamu vyberu slovo s největší shodou `best_match = sorted(word_scores, key=lambda item: item[1])[0][0]`. Textové výsledky jsou ve formátu `vybrané slovo - [(slovo - % shoda)]`.

### Výstup pro Signal1.txt

![Graf slov v signálu 1.](img/signal1.png)

```text
Odhadnutá věta pro Signal1.txt:
the make method the time make for prepare the solution

[('the', 1875.1), ('for', 2531.94), ('give', 2888.84), ('make', 2899.82)]
[('make', 1191.61), ('give', 1792.11), ('work', 2629.78), ('method', 2884.91)]
[('method', 2332.04), ('lecture', 2440.93), ('long', 2664.14), ('fail', 2750.52)]
[('the', 1908.39), ('for', 2585.34), ('make', 2796.73), ('no', 2933.21)]
[('time', 1350.49), ('for', 2221.56), ('fail', 2599.89), ('probable', 2981.92)]
[('make', 2426.98), ('method', 2438.23), ('for', 2457.57), ('long', 2619.12)]
[('for', 2412.41), ('time', 2967.9), ('give', 3330.13), ('probable', 3483.2)]
[('prepare', 2490.24), ('idea', 2712.7), ('for', 3009.15), ('lecture', 3233.25)]
[('the', 1972.75), ('no', 2141.31), ('make', 2443.81), ('give', 2565.27)]
[('solution', 3984.78), ('for', 4802.29), ('probable', 5068.32), ('succeed', 5116.79)]
```

### Výstup pro Signal2.txt

![Graf slov v signálu 2.](img/signal2.png)

```text
Odhadnutá věta pro Signal2.txt:
the give give for for

[('the', 1292.28), ('make', 2822.48), ('no', 2944.51), ('give', 3163.48)]
[('give', 2461.78), ('make', 2526.53), ('for', 2702.41), ('method', 3116.93)]
[('give', 2493.73), ('make', 2539.34), ('no', 3202.92), ('easy', 3396.55)]
[('for', 2626.03), ('the', 3025.63), ('time', 3347.29), ('give', 3394.13)]
[('for', 2896.96), ('give', 4545.17), ('time', 4596.33), ('the', 4786.74)]
```

### Výstup pro Signal3.txt

![Graf slov v signálu3.](img/signal3.png)

```text
Odhadnutá věta pro Signal3.txt:
the for for lecture lecture

[('the', 2424.85), ('for', 2789.68), ('make', 2933.73), ('no', 2964.64)]
[('for', 2506.28), ('time', 3477.01), ('give', 3841.06), ('the', 3905.52)]
[('for', 1969.88), ('give', 2393.84), ('the', 2717.08), ('make', 2782.96)]
[('lecture', 3672.31), ('is', 4228.02), ('the', 4352.5), ('make', 4489.82)]
[('lecture', 4186.63), ('idea', 4672.23), ('succeed', 5155.02), ('method', 5272.61)]
```

### Závěr

Tento přístup poskytuje efektivní metodu pro analýzu a detekci slov v rozdělenou a pomalu mluvenou řečí, ovšem u "normální" řeči má velké problémy ať už slova správně určit, tak i větu správně rozdělit. Algoritmus se dá určitě vylepšit, já bych ovšem ale volil například strojové učení pro další zpracování podobných dat.

### Použité zdroje

- [Numpy Dokumentace](https://numpy.org/doc/)
- [Librosa dokumentace](https://librosa.org/doc/latest/index.html)
- [Matplotlib dokumentace](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)
- [Pyttsx3 dokumentace](https://pyttsx3.readthedocs.io/en/latest/)
