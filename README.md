# 🏏 Dream11 Team Recommendation System using Machine Learning & Explainable AI

> An explainable AI-powered fantasy cricket platform that combines machine learning, optimization algorithms, and conversational AI to generate transparent, data-driven Dream11 team recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![ML](https://img.shields.io/badge/Machine%20Learning-CatBoost-orange)
![XAI](https://img.shields.io/badge/Explainable-AI-purple)
![Optimization](https://img.shields.io/badge/ILP-PuLP-red)

---

## 🎯 Overview

Fantasy sports platforms have traditionally relied on black-box prediction systems that provide recommendations without explaining the reasoning behind them.

This project takes a fundamentally different approach.

Instead of merely predicting player performance, we built an intelligent decision-support system that helps users understand **why** a player is recommended, **how** the recommendation was generated, and **what strategic trade-offs exist** when constructing a fantasy team.

The platform combines:

* Advanced machine learning models for player performance prediction
* Explainable AI (XAI) techniques for transparency
* Integer Linear Programming (ILP) for mathematically optimal team selection
* Conversational AI for natural-language explanations
* Interactive analytics and visualization tools

The result is a fantasy sports assistant that functions more like a cricket strategist than a simple prediction engine.

---

# ✨ Key Features

### 🧠 Explainable AI Recommendations

Unlike traditional fantasy sports predictors, every recommendation can be traced back to measurable player attributes and historical performance patterns.

Users can:

* View feature importance explanations
* Understand prediction drivers
* Compare players visually
* Ask natural language questions about recommendations

---

### 📊 Advanced Performance Forecasting

Predicts fantasy points across:

* Batting
* Bowling
* Fielding

using ensemble machine learning models trained on ball-by-ball cricket data.

Supported models include:

* Random Forest
* XGBoost
* CatBoost
* Gradient Boosting

CatBoost emerged as the best-performing architecture across all tasks.

---

### ⚙️ Optimal Team Generation

A custom Integer Linear Programming engine generates mathematically optimal Dream11 teams while respecting:

* Budget constraints
* Team composition rules
* Player role requirements
* Captain/Vice-Captain logic
* User-defined strategies

---

### 🎲 Risk-Aware Team Building

Users can choose between:

* Stable Teams
* Balanced Teams
* High-Risk Teams

using a custom variance-based optimization framework.

This enables personalized strategies based on risk appetite.

---

### 🤖 Conversational AI Assistant

An integrated LLM assistant allows users to ask:

> Why was Player A selected?

> Why is Player B captain?

> Which player has the highest upside?

> What makes this a low-risk team?

and receive contextual explanations grounded in model outputs.

---

## 🏗 System Architecture

```text
                ┌──────────────────────┐
                │     Frontend UI      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Backend API Layer  │
                └──────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼

 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
 │ ML Inference│   │ ILP Optimizer│   │ LLM Assistant│
 └─────────────┘   └──────────────┘   └──────────────┘

        ▲
        │

 ┌─────────────────────┐
 │ Feature Engineering │
 └─────────────────────┘

        ▲
        │

 ┌─────────────────────┐
 │ Cricsheet Datasets  │
 └─────────────────────┘
```

---

## 📈 Data Pipeline

The system automatically processes ball-by-ball match data from Cricsheet.

### Supported Formats

* T20
* ODI
* Test

### Processing Stages

1. Automated data ingestion
2. Match chronology reconstruction
3. Fantasy point calculation
4. Feature engineering
5. Player profiling
6. Model training
7. Team optimization

---

## 🔬 Feature Engineering

Over 100 engineered features are generated for every player.

### Career Statistics

* Career runs
* Career wickets
* Batting average
* Economy rate

### Form Metrics

* Last 3 matches
* Last 10 matches
* Momentum score

### Matchup Intelligence

* Historical performance vs opponents
* Team-specific trends
* Venue-adjusted metrics

### Risk Metrics

* Performance variance
* Consistency scores
* Volatility indicators

### Fantasy-Specific Features

* Simulated player cost
* Role identification
* Captaincy suitability

---

## 📊 Model Performance

### Batting Prediction

| Model         | MAE   | R²    |
| ------------- | ----- | ----- |
| Random Forest | 1.193 | 0.973 |
| XGBoost       | 1.138 | 0.979 |
| CatBoost      | 1.036 | 0.981 |

### Bowling Prediction

| Model         | MAE   | R²    |
| ------------- | ----- | ----- |
| Random Forest | 4.911 | 0.961 |
| XGBoost       | 3.948 | 0.978 |
| CatBoost      | 3.648 | 0.983 |

### Fielding Prediction

| Model         | MAE   | R²    |
| ------------- | ----- | ----- |
| Random Forest | 0.420 | 0.938 |
| XGBoost       | 0.270 | 0.962 |
| CatBoost      | 0.252 | 0.967 |

🏆 CatBoost was selected as the final production model due to its superior predictive performance across all three tasks.

---

## ⚡ Optimization Engine

The team selection problem is formulated as an Integer Linear Programming problem.

Objective:

Maximize expected fantasy points while satisfying:

* Budget ≤ 100 credits
* Team size = 11
* Role constraints
* Team composition constraints
* Captain and Vice-Captain selection
* Risk profile requirements

The optimization layer guarantees mathematically optimal team recommendations.

---

## 🎨 User Experience

The platform includes:

### Team Builder

* Match selection
* Team selection
* Budget management
* Role balancing

### Analytics Dashboard

* Player radar charts
* Statistical comparisons
* Performance trends

### Explainability Dashboard

* Feature importance visualization
* Prediction breakdowns
* Strategy insights

### AI Assistant

* Natural language explanations
* Team analysis
* Recommendation justification

---

## 🛠 Tech Stack

### Machine Learning

* Scikit-Learn
* CatBoost
* XGBoost
* Random Forest

### Optimization

* PuLP
* CBC Solver

### Backend

* FastAPI
* Python

### Frontend

* Next.js
* React
* Chart.js

### AI

* Groq
* Llama 3
* Explainable AI

### Data

* Pandas
* NumPy
* Cricsheet

---

## 🚀 Future Work

* Live match integration
* IPL auction simulation
* Reinforcement learning-based strategy optimization
* Multi-sport fantasy support
* Real-time captaincy recommendations
* Personalized recommendation agents

---

## 👥 Team

Developed by Team 2, Cynaptics Club
Indian Institute of Technology Indore

---

## 📜 License

This project is released under the MIT License.

---

## ⭐ Philosophy

Most fantasy sports systems tell users what to do.

This project aims to tell users **why**.

By combining explainable machine learning, mathematical optimization, and conversational AI, we transform fantasy team generation from a prediction problem into a strategic decision-making experience.
