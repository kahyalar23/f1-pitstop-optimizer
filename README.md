<div align="center">

# 🏎️ F1 Pit Stop Optimizer

**Physics-grounded simulation & reinforcement learning engine for Formula 1 pit stop strategy**

**Fizik tabanlı simülasyon ve pekiştirmeli öğrenme ile Formula 1 pit stop stratejisi optimizasyonu**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Notebooks](https://img.shields.io/badge/Notebooks-4-F37626?style=flat-square&logo=jupyter&logoColor=white)](notebooks/)

</div>

---

## 📖 English

### Overview

This project builds a multi-layered engineering simulation of a Formula 1 pit stop — from individual crew member task timing to full race strategy optimization via Reinforcement Learning.

The goal is not just simulation, but **grounded simulation**: every model is derived from first principles, calibrated against real FastF1 telemetry data, and statistically validated.

### What's Inside

| Module | Description | Key Technique |
|--------|-------------|---------------|
| `simulation/monte_carlo.py` | Stochastic pit stop duration model | Log-Normal sampling, Critical Path Method |
| `simulation/tire_model.py` | Tyre grip & degradation | Pacejka Magic Formula, thermal ODE |
| `simulation/human_factors.py` | Crew fatigue & error probability | Osman-Sheridan model, THERP, Yerkes-Dodson |
| `strategy/environment.py` | Full race Gym environment | OpenAI Gymnasium (8-dim state, 4 actions) |
| `strategy/rl_agent.py` | Race strategy optimization | PPO with GAE (stable-baselines3) |
| `data/fastf1_loader.py` | Real telemetry ingestion + validation | FastF1 API, KS-test, MAE, coverage probability |
| `api/main.py` | REST API for all modules | FastAPI |

### Core Mathematics

**1. Pit Stop Duration (Critical Path)**

Each wheel station runs in parallel. The slowest corner determines the total time:

$$T_{pit} = T_{reaction} + \max_{i \in \{FL,FR,RL,RR\}}(T_i) + T_{jack}$$

Each sub-task sampled from a Log-Normal distribution (always positive, right-skewed — physically correct):

$$t_{task} \sim \text{LogNormal}(\mu_{\ln},\ \sigma_{\ln}), \quad \mu_{\ln} = \ln\mu - \tfrac{1}{2}\ln\!\left(1+\tfrac{\sigma^2}{\mu^2}\right)$$

**2. Pacejka Magic Formula (Tyre Grip)**

$$\mu(x) = D \cdot \sin\!\left(C \cdot \arctan\!\left(Bx - E(Bx - \arctan(Bx))\right)\right) \cdot e^{-\lambda E_{cum}}$$

**3. Crew Fatigue (Osman-Sheridan)**

$$F(n) = F_{max}\!\left(1 - e^{-\beta_{eff} \cdot W_n}\right), \quad P_{error}(F) = P_{base} \cdot e^{\gamma F}$$

**4. PPO Strategy Agent (Clipped Objective)**

$$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),1{-}\epsilon,1{+}\epsilon)\hat{A}_t\right)\right]$$

### Notebooks

Four step-by-step Jupyter notebooks — each explains the math, implements the model, and visualizes results:

| # | Notebook | Topics |
|---|----------|--------|
| 01 | [Monte Carlo Simulation](01_monte_carlo_simulation.ipynb) | Log-Normal derivation, CPM Gantt, crew comparison, sensitivity analysis |
| 02 | [Tire Degradation Model](02_tire_degradation_model.ipynb) | Pacejka components, thermal ODE, wear cliff, undercut heatmap |
| 03 | [Human Factors Model](03_human_factors_model.ipynb) | Fatigue accumulation, Yerkes-Dodson curve, THERP error types, 18-crew analysis |
| 04 | [RL Strategy & Validation](04_rl_strategy_validation.ipynb) | PPO objective, race environment, KS-test vs real F1 data |

### Quickstart

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/f1-pitstop-optimizer.git
cd f1-pitstop-optimizer

# Tüm paketleri tek komutla kur
pip install -r requirements.txt

# Run Monte Carlo simulation
python -m pitstop.simulation.monte_carlo --crew elite --iterations 10000 --compare

# Simulate a tyre stint
python -m pitstop.simulation.tire_model

# Run fatigue simulation
python -m pitstop.simulation.human_factors

# Start FastAPI backend
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# Train RL agent (requires stable-baselines3 + torch)
python -m pitstop.strategy.rl_agent train --race monza --steps 500000
```

### Validation

Simulation outputs are statistically validated against real FastF1 telemetry:

- **Kolmogorov-Smirnov test** — does the simulated distribution match real pit stop times?
- **Mean Absolute Error** — point estimate accuracy (target: < 0.15s)
- **Coverage probability** — does the P5–P95 interval capture 90% of real events?
- **Tyre degradation MAE** — lap time prediction error per compound per age bin

### Project Structure

```
f1-pitstop-optimizer/
├── 01_monte_carlo_simulation.ipynb
├── 02_tire_degradation_model.ipynb
├── 03_human_factors_model.ipynb
├── 04_rl_strategy_validation.ipynb
├── pitstop/
│   ├── simulation/
│   │   ├── monte_carlo.py        # Stochastic pit stop timing
│   │   ├── tire_model.py         # Pacejka + thermal degradation
│   │   └── human_factors.py      # Fatigue & cognitive error model
│   ├── strategy/
│   │   ├── environment.py        # OpenAI Gym race environment
│   │   └── rl_agent.py           # PPO strategy agent
│   └── data/
│       └── fastf1_loader.py      # Real telemetry + validation
├── api/
│   └── main.py                   # FastAPI REST backend
├── requirements.txt
└── setup.py
```

### Dependencies

```
numpy · scipy · pandas · matplotlib     # Scientific computing
fastf1                                  # Real F1 telemetry
gymnasium · stable-baselines3 · torch   # Reinforcement learning
fastapi · uvicorn                       # REST API
jupyter · plotly                        # Notebooks & visualization
```

### License

MIT License — free to use, modify, and distribute with attribution.

---

## 📖 Türkçe

### Genel Bakış

Bu proje, bir Formula 1 pit stop'unun çok katmanlı mühendislik simülasyonunu inşa eder — bireysel ekip üyelerinin görev sürelerinden, Pekiştirmeli Öğrenme ile tam yarış stratejisi optimizasyonuna kadar.

Amaç yalnızca simülasyon değil, **temellendirilmiş simülasyon**: her model birinci prensiplerden türetilmiş, gerçek FastF1 telemetri verisine göre kalibre edilmiş ve istatistiksel olarak doğrulanmıştır.

### Neler Var

| Modül | Açıklama | Temel Teknik |
|-------|----------|--------------|
| `simulation/monte_carlo.py` | Stokastik pit stop süre modeli | Log-Normal örnekleme, Kritik Yol Yöntemi |
| `simulation/tire_model.py` | Lastik tutuş & degradasyon | Pacejka Magic Formula, termal ODE |
| `simulation/human_factors.py` | Ekip yorgunluğu & hata olasılığı | Osman-Sheridan modeli, THERP, Yerkes-Dodson |
| `strategy/environment.py` | Tam yarış Gym ortamı | OpenAI Gymnasium (8 boyutlu state, 4 aksiyon) |
| `strategy/rl_agent.py` | Yarış stratejisi optimizasyonu | PPO + GAE (stable-baselines3) |
| `data/fastf1_loader.py` | Gerçek telemetri + validasyon | FastF1 API, KS-testi, MAE, kapsama olasılığı |
| `api/main.py` | Tüm modüller için REST API | FastAPI |

### Temel Matematik

**1. Pit Stop Süresi (Kritik Yol)**

Her köşe paralel çalışır. En yavaş köşe toplam süreyi belirler:

$$T_{pit} = T_{reaksiyon} + \max_{i \in \{FL,FR,RL,RR\}}(T_i) + T_{kriko}$$

Her alt görev Log-Normal dağılımdan örneklenir (her zaman pozitif, sağa çarpık — fiziksel olarak doğru).

**2. Pacejka Magic Formula (Lastik Tutuşu)**

Kayma enerjisi ile grip katsayısı arasındaki doğrusal olmayan ilişki.

**3. Yorgunluk Modeli (Osman-Sheridan)**

Kümülatif iş yükü ile üstel yorgunluk birikimi; yorgunluk arttıkça hata olasılığı üstel büyür.

**4. PPO Strateji Ajanı**

Her turda hangi compound ile pit yapılacağına veya devam edileceğine karar veren derin RL ajanı.

### Notebook'lar

Dört adım adım Jupyter notebook — her biri matematiği açıklar, modeli uygular ve sonuçları görselleştirir:

| # | Notebook | Konular |
|---|----------|---------|
| 01 | Monte Carlo Simülasyonu | Log-Normal türetimi, CPM Gantt, ekip karşılaştırması, duyarlılık analizi |
| 02 | Lastik Degradasyon Modeli | Pacejka bileşenleri, termal ODE, wear cliff, undercut ısı haritası |
| 03 | İnsan Faktörleri Modeli | Yorgunluk birikimi, Yerkes-Dodson eğrisi, THERP hata tipleri, 18 kişilik ekip analizi |
| 04 | RL Strateji & Validasyon | PPO hedef fonksiyonu, yarış ortamı, gerçek F1 verisiyle KS-testi |

### Hızlı Başlangıç

```bash
# Kur ve yükle
git clone https://github.com/KULLANICI_ADIN/f1-pitstop-optimizer.git
cd f1-pitstop-optimizer

# Tüm paketleri tek komutla kur
pip install -r requirements.txt

# Monte Carlo simülasyonu çalıştır
python -m pitstop.simulation.monte_carlo --crew elite --iterations 10000 --compare

# FastAPI backend başlat
uvicorn api.main:app --reload
# → http://localhost:8000/docs

# RL ajanı eğit
python -m pitstop.strategy.rl_agent train --race monza --steps 500000
```

### Doğrulama (Validasyon)

Simülasyon çıktıları gerçek FastF1 telemetri verisiyle istatistiksel olarak doğrulanmaktadır:

- **Kolmogorov-Smirnov testi** — simüle edilen dağılım gerçek pit stop süresiyle uyuşuyor mu?
- **Ortalama Mutlak Hata** — nokta tahmini doğruluğu (hedef: < 0.15s)
- **Kapsama olasılığı** — P5–P95 aralığı gerçek olayların %90'ını kapsıyor mu?
- **Lastik degradasyon MAE** — compound ve lastik yaşı başına tur süresi tahmin hatası

### Bağımlılıklar

```
numpy · scipy · pandas · matplotlib     # Bilimsel hesaplama
fastf1                                  # Gerçek F1 telemetrisi
gymnasium · stable-baselines3 · torch   # Pekiştirmeli öğrenme
fastapi · uvicorn                       # REST API
jupyter · plotly                        # Notebook ve görselleştirme
```

---

<div align="center">

**Bu proje, gerçek dünya mühendislik problemlerinde fizik modelleme,**
**stokastik simülasyon ve makine öğrenmesinin bir arada nasıl uygulandığını göstermektedir.**

*This project demonstrates how physics modeling, stochastic simulation,*
*and machine learning can be applied together to real-world engineering problems.*

</div>
