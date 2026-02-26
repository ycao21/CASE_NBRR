CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation

CASE models next-basket repurchase by:

- Representing each item’s purchase history as a **calendar-time binary signal**
- Extracting cadence-aware features using **multi-scale temporal CNN**
- Modeling cross-item dependencies with **Induced Set Attention (ISAB)**
- Producing per-item repurchase scores via an MLP ranking layer

The implementation is written in PyTorch and supports training on public datasets of Instacart, TaFeng and Dunnhumby DC by running

```
python -m src.runner --config configs/train_config.yaml --local
```
