# 🚚 DelivTrack 📦 🗺️
A Streamlit web app that visualizes and simulates optimized delivery routes from a Kaggle dataset using a greedy nearest-neighbor approach.
DelivTrack designed to demonstrate efficient order selection and route mapping for delivery systems.

## 📁 Project Directory Structure 🧠💬

```
DelivTrack/
├── app.py                     # 🖥️🎨 Streamlit frontend
├── logic.py                   # ⚙️ Backend logic for route and order selection
├── train.csv                  # 📊 Local dataset 
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
git clone https://github.com/SankalpBankar/DelivTracker.git
cd DelivTracker
```

### 2️⃣ Install Dependencies 📦
```sh
pip install -r requirements.txt
```


### 3️⃣ Set Up Environment Variables 🔑
Create a .env file in the root directory
Add your API key:
```sh
GROQ_API_KEY=groq_api_key
```

### 4️⃣ Load the LLM (LLaMA3.1) 🦙
The code uses LangChain + Groq to load the model:
```sh
groq_api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    groq_api_key=groq_api_key,  
    model_name="llama-3.1-8b-instant",
    temperature=0.2        
)
```
