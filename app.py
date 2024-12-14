import streamlit as st

from ml_app import run_ml_app

def main():
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Penguin Analysis")
        st.write("This is a tool to classify penguin species based on penguin body measurements.")
    elif choice == "Machine Learning":
        run_ml_app()


if __name__ == '__main__':
    main()