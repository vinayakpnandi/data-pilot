# DataPilot

> AI-powered data analysis assistant that lets users upload datasets and ask questions in natural language.

DataPilot is a local AI Data Analyst built using **Python, Streamlit, Pandas, Ollama, and Qwen 3 1.7B**.

Users can upload a CSV or Excel dataset and ask questions such as:

- What is the average revenue?
- Which title has the highest revenue?
- What is the total sales?
- Which category performs best?
- How many records are there?

The application analyzes the dataset and returns concise, human-readable answers.

---

## 🚀 Features

- 📁 Upload CSV and Excel datasets
- 💬 Ask questions using natural language
- 📊 Analyze data using Pandas
- 🤖 Local AI using Qwen 3 1.7B
- ⚡ Hybrid AI + deterministic data analysis
- 🔒 Local processing through Ollama
- 📈 Basic data analysis and comparisons
- 🌐 Simple Streamlit interface

---

## 🧠 Architecture

```text
User
 │
 ▼
Streamlit
 │
 ▼
Dataset
 │
 ▼
Pandas DataFrame
 │
 ▼
Analyst Agent
 │
 ├── Pandas Analysis
 │
 └── Qwen 3 1.7B
        │
        ▼
     Answer
````

Pandas handles numerical calculations and deterministic operations, while Qwen is used for natural-language understanding when required.

---

## 🛠️ Tech Stack

| Technology  | Purpose          |
| ----------- | ---------------- |
| Python      | Core development |
| Streamlit   | User interface   |
| Pandas      | Data analysis    |
| Qwen 3 1.7B | Local AI         |
| Ollama      | LLM runtime      |
| OpenPyXL    | Excel support    |
| Plotly      | Visualization    |

---

## 📂 Project Structure

```text
data-pilot/
│
├── app.py
├── analyst_agent.py
├── data_tools.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/vinayakpnandi/data-pilot.git
cd data-pilot
```

### 2. Create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Install Qwen using Ollama

```powershell
ollama pull qwen3:1.7b
```

Verify:

```powershell
ollama list
```

### 5. Run the application

```powershell
streamlit run app.py
```

---

## 💬 Example Questions

```text
What is the average revenue?
```

```text
Which title has the highest revenue?
```

```text
What is the total revenue?
```

```text
Which category performs best?
```

```text
How many records are there?
```

---

## 🔮 Future Improvements

* Automatic chart generation
* Advanced statistical analysis
* Better conversational memory
* More complex analytical questions
* PDF/document analysis
* RAG-based document question answering
* Automated business insights

---

## 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Python
* Pandas
* Data Analysis
* Streamlit
* Local LLMs
* Ollama
* Qwen
* AI Agent Development
* Natural-language data interaction

---

## 👨‍💻 Author

**Vinayak Prakash Nandi**

Data Science Engineering Student
New Horizon College of Engineering

GitHub: [@vinayakpnandi](https://github.com/vinayakpnandi)


This version is much more appropriate for your GitHub portfolio: **clear, professional, and not overloaded.**
```
