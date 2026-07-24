# FlowDraft: итог исследовательской сессии

**Дата среза:** 24 июля 2026.  
**Цель:** восстановить отсутствующий training pipeline для Orthrus на
`Qwen/Qwen3-1.7B`, затем проверить гипотезу FlowDraft: может ли обучаемый
categorical Flow Map drafter повысить число токенов, принимаемых lossless
AR-verifier, и тем самым ускорить декодирование.

Это журнал фактически выполненной работы и сохраненных артефактов. Он отделяет
строгие результаты от ранних диагностических запусков.

## Короткий итог

1. Создан воспроизводимый pipeline обучения и строгой greedy-оценки Orthrus
   поверх официального inference-репозитория. Он запускается на одной A100,
   сохраняет `best` и `last`, записывает все метрики и архивирует веса в
   приватный Hugging Face Hub.
2. Реализирована и проверена lossless схема FlowDraft/EagleFlow: параллельный
   endpoint Flow Map предлагает блок токенов, а frozen Qwen AR verifier
   принимает только совпадающий префикс и при отказе сам выдает следующий
   токен. На текущих AIME25 и HumanEval prompt-наборах greedy parity равна
   **100%**.
3. Лучший наш метод, EagleFlow, дает честное ускорение относительно AR:
   **1.493x на AIME25** и **1.394x на HumanEval**. Это лучше или сопоставимо с
   заново обученным бюджетно-сопоставимым Orthrus, но существенно ниже
   опубликованного Orthrus checkpoint (3.124x и 4.090x в том же локальном
   протоколе).
4. Главный научный вывод пока отрицательный, но полезный: многие более
   сложные CFM-варианты не увеличили end-to-end скорость. Ускорение требует не
   просто хорошей локальной flow-loss, а дешевого параллельного proposal path,
   высокого качества **первых** токенов блока и корректной общей стоимости
   proposal+verification.

## Что было восстановлено для Orthrus

Официальный репозиторий Orthrus предоставляет архитектуру и inference, но не
публичный training pipeline. В этом репозитории добавлены:

- подготовка `nvidia/Nemotron-Post-Training-Dataset-v2` в packed
  последовательности длины 2,048;
- обучение только `q_proj_diff`, `k_proj_diff`, `v_proj_diff` при frozen
  Qwen3-1.7B backbone;
- block-causal attention: чистый AR context до anchor и bidirectional
  attention внутри diffusion block;
- исходная Orthrus objective -- forward-KL от frozen AR teacher;
- fixed holdout, выбор `best` по greedy prefix acceptance, `last` в конце
  обучения, компактные trainer state без накопления numbered checkpoints;
- строгий greedy benchmark, где ускоренный текст посимвольно по токенам
  сравнивается с текстом обычного AR decoding;
- DataSphere/Vast launchers, preflight GPU/disk, сохранение протокола и
  автоматический upload `best`/`last` на Hugging Face Hub.

Практические проблемы окружения тоже закрыты: isolated virtualenv без системного
`ensurepip`, совместимые Python/PyTorch зависимости, gated Hugging Face data,
кэш на writable storage и checkpoints в `/dev/shm`, чтобы не переполнять
домашний 10-GB volume.

## Данные и разделение

Для основных контролируемых запусков использовался
`nvidia/Nemotron-Post-Training-Dataset-v2` с доменами `chat`, `math`, `code`.

| Назначение | Объем | Примечание |
|---|---:|---|
| Train pool | 49,750 packed sequences | 2,048 токенов на sequence |
| Holdout | 2,048 packed sequences | content-disjoint от train |
| Вычислительный бюджет основного run | 20,000 optimizer updates | batch size 1, 64 anchor blocks |

В первых quick-запусках train и eval могли пересекаться из-за streaming
dataset, поэтому они сохранены только как инженерные smoke-проверки. В
финальных EagleFlow и Orthrus run split проверяется до обучения. Поиск не нашел
ни одного текста benchmark prompt в 40.96M packed training tokens.

## Что именно представляет собой текущий FlowDraft

Текущая лучшая версия называется **EagleFlow**. Это не полная независимая
generative model и не второй AR decoder.

1. Frozen parent предоставляет exact AR context и features для следующего
   блока `K=32`.
2. Параллельная attention-conditioned endpoint Flow Map head предсказывает
   endpoint hidden-state trajectory и token embeddings для всего блока.
3. Frozen LM head переводит эти состояния в candidate tokens.
4. Обычный causal Qwen pass проверяет candidate block. Совпавший префикс
   принимается, а первый несовпавший токен берется у AR verifier. Поэтому
   greedy output идентичен базовой модели независимо от точности drafter.

Для длинного continuation run head был warm-started от короткого
`eagleflow_parallel_refine_3000_r1`, а frozen parent был
`flowdraft_v5_prefix_ecld_2000_r3/best`. Это следует явно учитывать: текущий
EagleFlow -- многостадийный метод, а не Flow Map, обученная с нуля на vanilla
Qwen за один запуск.

### Финальная функция потерь EagleFlow

Для каждого блока оптимизировалась

```text
L = 0.10 L_hidden + 0.10 L_embedding
  + 1.25 L_prefix
  + 0.05 L_diagonal + 0.02 L_consistency.
```

- `L_hidden`: RMS-нормированная MSE прогнозируемых и teacher hidden states;
- `L_embedding`: MSE в пространстве token embeddings;
- `L_prefix`: prefix-survival cross-entropy с геометрическим decay `0.85`;
  ранние позиции имеют больший вес, так как только непрерывный начальный
  префикс дает ускорение;
- `L_diagonal`: teacher-matching на случайном flow time (вероятность 0.10);
- `L_consistency`: endpoint/self-consistency между diagonal и endpoint path.

В лучшем continuation `teacher_forcing=0`, `feedback_mode=continuous`; на
inference используется один параллельный jump. Soft-KL к логитам в этой
конкретной конфигурации выключен (`kl_loss_weight=0`), а не скрыт внутри
метрики.

## Хронология экспериментов

Ниже приведены сохраненные серии. Ранние таблицы основаны на десяти sanity
prompts и не должны сравниваться с AIME25/HumanEval или между собой как
paper-grade ranking.

| Семейство | Идея | Итог строгого FP32 sanity benchmark | Вывод |
|---|---|---|---|
| CFM v2 / ранний FlowDraft | Categorical endpoint transport + ECLD | pilot speedup 1.341x | Доказал работоспособность path, но был слишком коротким и нестабильным. |
| FlowDraft v3 | verifier-aligned endpoint objective | 1.196x | Небольшой gain, недостаточный для цели. |
| FlowDraft v4 | teacher-forced endpoint path | 1.051x | Диагностический запуск; transfer на generation слабый. |
| FlowDraft v5 | prefix-aware ECLD | 1.296x | Лучший из ранних родителей; стал базой EagleFlow. |
| FlowTree | tree/verifier-aware proposal | training-only diagnostic | Не стал выбранным inference path. |
| R2Flow | fixed-point residual corrector, имитация `J^2` | 1.066x, TPF 1.170 | Прокси на holdout не перенеслась на независимые prompts. |
| CacheFlow | one-pass final hidden trajectory | 0.916x, TPF 1.064 | Lossless, но почти не принимал токены. |
| HydraFlow | self-conditioned sequential latent rollout | 0.698x, TPF 1.078 | Python-level sequential rollout съел выигрыш. |
| Local Simplex CFM | flow на top-128 support | 1.049x, TPF 1.153 | Support coverage около 81.6%; пропущенные teacher tokens нельзя восстановить. |
| Rescue / dynamic support CFM | train-derived или retrieved support | 1.022-1.049x | Расширение support не перенеслось и добавило overhead. |
| EagleFlow attention screen | attention-conditioned endpoint head | 0.365x | Ранняя последовательная схема была слишком дорогой. |
| EagleFlow parallel | один параллельный endpoint proposal | AIME25 1.493x; HumanEval 1.394x | Текущий лучший FlowDraft результат. |

Все отрицательные результаты с конфигурациями и сырыми метриками сохранены в
[`experiments/`](/Users/yaroslavpelehov/Downloads/FlowDraft/experiments).

## Главные сравнимые результаты

Один и тот же строгий runtime protocol используется для трех строк ниже:
FP32, eager attention, greedy decoding, 128 generated-token cap, одинаковые
версионированные prompts. `Parity` -- точное token-by-token совпадение с
sequential AR output.

| Метод | AIME25 TPF | AIME25 speedup | AIME25 parity | HumanEval TPF | HumanEval speedup | HumanEval parity |
|---|---:|---:|---:|---:|---:|---:|
| FlowDraft, EagleFlow endpoint Flow Map | 1.761 | **1.493x** | 100% (30/30) | 1.647 | **1.394x** | 100% (164/164) |
| Orthrus, наш budget-matched reconstruction | 1.371 | 1.276x | 100% (30/30) | 1.591 | 1.476x | 100% (164/164) |
| Orthrus, released `chiennv/Orthrus-Qwen3-1.7B` | 3.459 | 3.124x | 100% | 4.559 | 4.090x | 100% |

`TPF = total generated tokens / total counted frozen-verifier forward passes`.
Speedup измеряет полное wall-clock время, включая drafter. TPF и speedup не
обязаны быть пропорциональны: kernel overhead, prefill и цена proposal path
различаются.

**Важно:** среднюю accepted length нельзя использовать как headline ranking
между EagleFlow и Orthrus. У Orthrus один proposal cycle содержит дополнительный
model pass, у EagleFlow -- lightweight external head. Одинаковая accepted length
может иметь разную цену. Ее стоит применять только как диагностику внутри
одного и того же decoder loop.

## Как тестировалось

### Losslessness

Для каждого prompt запускаются AR и accelerated greedy decoding, затем
сравниваются все generated token ids. Если хотя бы один токен различается,
benchmark с `--require-parity` завершается ошибкой. В финальных таблицах все
194 prompt-а прошли проверку. Это демонстрирует losslessness только для
**greedy `T=0`** режима.

### Prompt sets

- `math-ai/aime25`: 30 prompts;
- `openai/openai_humaneval`: 164 prompts.

Они подготовлены и закоммичены вместе с revision metadata в `paper_eval/`.
На этих запусках **не считались** AIME answer accuracy и HumanEval `pass@1`:
это efficiency/losslessness benchmark на официальных task prompts, а не полная
задачная оценка из статьи.

### Что еще не закрыто до paper-grade reproduction

- В статье Orthrus сравнивается на более широком suite: GSM8K, MATH-500,
  AIME24, AIME25, HumanEval, MBPP, Pseudo2code и LiveCodeBench-v5.
- Нужны official task scoring, sampling `T=1` с rejection sampling и
  losslessness этого режима.
- Нужны несколько запусков/порядков benchmark-а и доверительные интервалы для
  wall-clock speedup.
- Тренировочный масштаб несопоставим с paper training: у авторов порядка
  1.92B token exposures на 8xH200; наш основной run -- 20k updates на одной
  A100 и 49,750 packed sequences.

Следовательно, текущий результат -- корректный локальный efficiency experiment
и baseline для дальнейших абляций, но не заявление о воспроизведении Table 1
статьи.

## Веса, код и как вернуться к результату

Код, конфигурации, prompt sets, raw per-prompt metrics и train curves находятся
в Git. Тензоры не коммитятся, чтобы не раздувать историю; для главных запусков
сохранены обе точки восстановления:

| Run | Hub path |
|---|---|
| EagleFlow `best` и `last` | `Yaroslav574389/FlowDraft-EagleFlow-Qwen3-1.7B/runs/eagleflow_parallel_continue_20000_r1/{best,last}` |
| Orthrus `best` и `last` | `Yaroslav574389/FlowDraft-EagleFlow-Qwen3-1.7B/runs/orthrus_budgetmatch_20000_r2/{best,last}` |

Точные metadata и пути лежат в соответствующих
`hf_upload_manifest.json`. Главные самостоятельные архивы:

- [`experiments/eagleflow_parallel_continue_20000_r1/README.md`](/Users/yaroslavpelehov/Downloads/FlowDraft/experiments/eagleflow_parallel_continue_20000_r1/README.md);
- [`experiments/orthrus_budgetmatch_20000_r2/README.md`](/Users/yaroslavpelehov/Downloads/FlowDraft/experiments/orthrus_budgetmatch_20000_r2/README.md);
- [`scripts/run_vast_eagleflow_paper_eval.sh`](/Users/yaroslavpelehov/Downloads/FlowDraft/scripts/run_vast_eagleflow_paper_eval.sh);
- [`scripts/run_vast_orthrus_budgetmatch.sh`](/Users/yaroslavpelehov/Downloads/FlowDraft/scripts/run_vast_orthrus_budgetmatch.sh).

## Практический вывод и следующий эксперимент

Метод уже имеет корректный lossless verifier и измеримый 1.39-1.49x gain, но
еще не подтвердил гипотезу, что CFM сам по себе заметно превосходит Orthrus на
этом масштабе. Следующий научно осмысленный шаг -- не добавлять еще один
дорогой inference pass. Нужно продолжать именно дешёвый parallel endpoint path
и проводить контролируемую абляцию: добавить soft AR-logit KL к текущему
prefix-weighted feature/endpoint objective, оставив architecture, data split,
compute budget и strict benchmark неизменными. Так можно изолировать вклад
distribution matching в качество первых токенов и не смешать его с новой
стоимостью decoder-а.

