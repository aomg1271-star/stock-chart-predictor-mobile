# Streamlit Cloud entry point. Keeps app.py as the single source of UI logic.
exec(open("app.py", encoding="utf-8").read())
