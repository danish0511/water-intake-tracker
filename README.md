# 💧 Water Intake Tracker

AI Water Tracker is a smart hydration monitoring system built using FastAPI, LangChain with LLaMA 3 via Groq, and SQLite. It allows users to log their daily water intake and get real-time hydration feedback powered by AI.

---

## 🚀 Features

- ✅ Log daily water intake via REST API
- 📊 Retrieve personal hydration history
- 🤖 Get intelligent hydration advice using Groq's llama-3.1-8b-instant model
- 🗃️ Store intake logs using SQLite
- 📝 Centralized logging for activity tracking

---

## 🧠 AI Integration

This project uses [LangChain](https://www.langchain.com/) with the [Groq API](https://groq.com/) to run LLaMA 3.1 (8B instant) models. It provides smart suggestions based on how much water a user has consumed.

---

## 📁 Project Structure

```
water_intake_tracker
├── src
    ├── agent.py
    ├── api.py
    ├── database.py
    ├── logger.py
    ├── water_tracker.db
├── dashboard.py  
├── requirements.txt
├── .env
├── app.log
├── water_tracker.db

```

---

## ⚙️ Setup Instructions

* Clone the repo
```
git clone https://github.com/your-username/ai-water-tracker.git
cd ai-water-tracker
```

* Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

* Install Dependencies
```
pip install -r requirements.txt
```

* Add a `.env` File
```
GROQ_API_KEY={your_groq_api_key_here}
DATABASE_URL=sqllite:///water_tracker.db
```

* Run the API Server
```
uvicorn src.api:app --reload
```

* Run the Streamlit Server
```
streamlit run dashboard.py
```
