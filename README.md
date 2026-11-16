# Nepali News Headline Generation using mBART

Abstractive headline generation for Nepali news articles. We fine-tuned
`facebook/mbart-large-50` on 86,628 scraped articles using 4-bit quantisation with LoRA, and
wrapped it in a Flutter app that takes an article and returns a headline.

The work was published in the *Journal of Science and Technology*:

> B. P. Paneru, M. Budhathoki, S. Panta, A. Bohora, D. Bhattarai and S. Bista,
> "Nepali News Headline Generation using mBART Model," *Journal of Science and Technology*,
> vol. 5, no. 1, pp. 23-29, 2026. ISSN 2362-1443 (print), 3059-9490 (online).

## What it does

You paste in the body of a Nepali news article and it writes a headline for it. Not an
extractive summary that picks an existing sentence out of the text, but a generated one, which
is why mBART is a reasonable choice here: it is pre-trained as a denoising autoencoder across
50 languages, so it already has some grasp of Nepali morphology before we show it a single
headline.

Nepali is low-resource for this task. There is no large headline corpus you can download, so
the first half of the project was building one.

## Dataset

The corpus is published on Kaggle:
**[Nepali News Dataset](https://www.kaggle.com/datasets/bibekpaneru1/news-dataset)** — around
86,600 records, 454 MB, a single CSV with three columns: `URL`, `news_article` and `headline`.

86,628 article/headline pairs scraped from Online Khabar, BBC Nepali and Ratopati. The headline
is the one a human editor actually wrote, which is what makes the pairs usable as supervision.

| Category | Articles |
|---|---|
| Economy | 35,360 |
| Sports | 22,168 |
| Entertainment | 19,160 |
| Health | 3,920 |
| Education | 2,960 |
| Opinion | 2,720 |

Split 90/10, so 77,965 for training and 8,663 for validation.

Cleaning removed HTML tags, special characters and symbols, and also stripped Latin letters
(a-z, A-Z) and Arabic numerals (0-9) out of the text, since those mostly came in as scraping
residue and the model was producing them in places where a Devanagari headline would not have
them. Tokenised with `MBart50TokenizerFast`, article truncated to 512 tokens and headline to 50.

The category distribution is heavily skewed towards economy and sports, which is a real
limitation; the model is noticeably better on those two than on opinion pieces.

If you are starting from the raw Kaggle file rather than our processed version, two things need
handling first. Some rows have a null or empty `news_article`, and because the scrape ran across
several portals and time windows there are duplicate URLs and repeated headlines in there, so
deduplicate before splitting or you will leak between train and validation. Many articles also
open with a dateline like `काठमाडौँ ।`, which you can keep or strip depending on what you are
doing with them.

## Training setup

| Parameter | Value |
|---|---|
| Model | `facebook/mbart-large-50` (~600M params) |
| Source / target language codes | `en_XX` / `ne_NP` |
| Max length, article / headline | 512 / 50 |
| Batch size, train / eval | 8 / 8 |
| Gradient accumulation | 2 |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Warmup steps | 250 |
| Quantisation | 4-bit |
| dtype | bfloat16 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |

Full fine-tuning a 600M-parameter model was not an option on the GPU we had, so the model is
loaded in 4-bit and only the LoRA adapters are trained. Three epochs because loss had mostly
flattened by then and the marginal gain was not worth the time on a shared Colab instance.

| Epoch | Training loss | Validation loss |
|---|---|---|
| 1 | 0.475100 | 0.454159 |
| 2 | 0.444300 | 0.441768 |
| 3 | 0.432600 | 0.437308 |

## Results

ROUGE against the human-written headlines, on a 1,000-article sample of the validation set:

| | Precision | Recall | F1 |
|---|---|---|---|
| ROUGE-1 | 0.4699 | 0.4736 | 0.4682 |
| ROUGE-2 | 0.3033 | 0.3037 | 0.3014 |
| ROUGE-L | 0.4671 | 0.4649 | 0.4545 |

And on a 2,000-article sample:

| | Precision | Recall | F1 |
|---|---|---|---|
| ROUGE-1 | 0.4372 | 0.4704 | 0.4428 |
| ROUGE-2 | 0.2625 | 0.2886 | 0.2671 |
| ROUGE-L | 0.4154 | 0.4465 | 0.4205 |

ROUGE-1 around 0.45 means roughly half the words in the generated headline also appear in the
editor's headline. The drop in ROUGE-2 (0.30 to 0.27) between the two samples is the honest
part of this: bigram overlap is where generated headlines fall apart, because the model picks
the right content words and then orders or inflects them differently from the editor. The
second sample is bigger and the scores are lower, so 0.4682 should be read as the top of the
range, not the number.

ROUGE also has a known problem for this task. A headline that is a perfectly good paraphrase
scores badly against a single reference, and there is only one reference per article here.

## The app

`headline_generator/` is a Flutter client. You type or paste article text, hit generate, and it
shows the headline with a copy button.

It talks to a FastAPI service that holds the model:

```
POST /generate_headline
Content-Type: application/json

{ "article": "..." }      ->      { "headline": "..." }
```

The endpoint URL is hardcoded in `lib/headlinegenerator.dart`. It is currently set to
`http://10.0.2.2:8000/generate_headline`, which is the Android emulator's route to localhost on
the host machine. Running on a real device over wifi means changing it to the host's LAN
address, and there is a commented line in the file showing that form.

**The FastAPI backend and the training notebook are not in this repo yet.** The app will build
and run, but it cannot generate anything without the service behind it. Getting those two
pushed is the next thing on the list.

Demo video: https://drive.google.com/file/d/1OOciR8UGa9iegeS8SraTPfa3gpbUZn7_/view?usp=drive_link

## Repo contents

```
headline_generator/          Flutter app (Android, iOS, web, desktop targets scaffolded)
major_projectv01.ipynb       leftover Whisper ASR notebook from a different project,
                             not related to headline generation
```

## Running the app

Needs the Flutter SDK (Dart comes with it). Tested on 3.0.2 and up.

```
git clone https://github.com/bibek0923/AI-based-Nepali-News-Headline-Generator.git
cd AI-based-Nepali-News-Headline-Generator/headline_generator
flutter pub get
flutter run
```

Point the URL in `lib/headlinegenerator.dart` at wherever your service is running first,
otherwise the generate button just returns an error string.

## Authors

Bibek Prasad Paneru, Mohit Budhathoki, Sumit Panta, Arudhi Bohora, Divya Bhattarai, and
Sharmila Bista (corresponding author), Department of Electronics and Computer Engineering,
National College of Engineering, IOE, Tribhuvan University.

## License

MIT.
