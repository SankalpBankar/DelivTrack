# 🚚 DelivTrack 📦 🗺️
A Streamlit web app that visualizes and simulates optimized delivery routes from a Kaggle dataset using a greedy nearest-neighbor approach.
DelivTrack designed to demonstrate efficient order selection and route mapping for delivery systems.

## 📁 Project Directory Structure 🧠💬

```
DelivTrack/
├── app.py                     # 🖥️🎨 Streamlit frontend
├── logic.py                   # ⚙️ Backend logic for route and order selection
├── df_orders.csv                  # 📊 Local dataset 
│
├── requirements.txt           # ✅ Dependencies for Streamlit app
├── README.md                  # 📖 Project overview and usage guide
│
└── .gitignore                 # 🛡️Ignore unnecessary files
```

## 💡 Tech Stack 🛠️
- **Python** 🐍 — Core programming language for logic and data handling
- **Streamlit** 🌐 — For building the interactive web interface
- **Pandas / NumPy** 📦 — Data handling
- **SciPy** ⚡ — Route distance optimization
- **Matplotlib** 📈 — Visualization

---

## ⚙️ Setup & Installation for DelivTrack 📦🗺️
Follow these steps to set up and track your order:
### 1️⃣ Clone the Repository 📥
```sh
git clone https://github.com/SankalpBankar/DelivTrack.git
cd DelivTrack
```

### 2️⃣ Install Dependencies 📦
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App 🚀
Ensure you have all backend files and Streamlit app ready. Then run:
```sh
streamlit run app.py
```

## 🛠️ Troubleshooting 🚨

### • Streamlit App not running
If the command fails or you get ModuleNotFoundError.
Ensure dependencies are installed and environment is active:
1. Activate your environment
```sh
source env/bin/activate
```
2. Re-install all dependencies
```sh
pip install -r requirements.txt
```

