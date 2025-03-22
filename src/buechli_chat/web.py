# src/buechli_chat/web.py
import sys
import streamlit.web.cli


def main():
    # Set sys.argv so that streamlit receives the target file argument.
    sys.argv = ["streamlit", "run", "src/buechli_chat/app.py"]
    streamlit.web.cli.main()


# Optional: run main if executed as a script
if __name__ == "__main__":
    main()
